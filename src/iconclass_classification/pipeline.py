"""Main pipeline orchestration for Iconclass classification."""

import json
import logging
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from iconclass_classification.filtering import (
    filter_children_only,
    sample_objects,
    validate_no_abb_in_sample,
)
from iconclass_classification.image_utils import (
    cache_image,
    compute_sha256,
    download_image,
    is_image_url,
    process_image,
)
from iconclass_classification.models import (
    ClassificationOptions,
    ClassificationRecord,
    IconclassCodeRank,
    IconclassDetails,
    ImageProcessingConfig,
    Metadata,
    MetadataObject,
    OllamaConfig,
    RunCounts,
    RunManifest,
    SamplingConfig,
    SubjectDetails,
)
from iconclass_classification.ollama_client import (
    USER_PROMPT,
    classify_image,
    extract_codes,
    save_classification_artifacts,
)

logger = logging.getLogger(__name__)


def setup_logging(log_file: Path) -> None:
    """Configure logging to file and console.

    Args:
        log_file: Path to log file
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


def get_git_commit() -> str | None:
    """Get current git commit SHA.

    Returns:
        Short commit SHA or None if not in git repository
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return None


def fetch_metadata(url: str, timeout: int = 30) -> dict[str, Any]:
    """Fetch metadata.json from URL.

    Args:
        url: URL to metadata.json
        timeout: Request timeout in seconds

    Returns:
        Raw metadata dictionary

    Raises:
        requests.RequestException: If fetch fails
    """
    logger.info(f"Fetching metadata from {url}")
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def create_run_directory(base_dir: Path) -> tuple[Path, str]:
    """Create timestamped run directory.

    Args:
        base_dir: Base directory for runs

    Returns:
        Tuple of (run_path, run_id)
    """
    now = datetime.now(UTC)
    run_id = now.strftime("%Y-%m-%dT%H-%M-%SZ")
    run_path = base_dir / run_id

    # Create subdirectories
    (run_path / "raw").mkdir(parents=True, exist_ok=True)
    (run_path / "data").mkdir(parents=True, exist_ok=True)
    (run_path / "classify").mkdir(parents=True, exist_ok=True)
    (run_path / "logs").mkdir(parents=True, exist_ok=True)
    (run_path / "results").mkdir(parents=True, exist_ok=True)

    logger.info(f"Created run directory: {run_path}")
    return run_path, run_id


def save_manifest(run_path: Path, manifest: RunManifest) -> None:
    """Save run manifest to JSON.

    Args:
        run_path: Run directory path
        manifest: Run manifest
    """
    manifest_path = run_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2))


def append_jsonl(file_path: Path, record: dict[str, Any]) -> None:
    """Append record to JSONL file.

    Args:
        file_path: Path to JSONL file
        record: Record to append
    """
    with open(file_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def select_image_url(obj: MetadataObject) -> str | None:
    """Select best image URL from object.

    Args:
        obj: Metadata object

    Returns:
        Selected image URL or None
    """
    # Prefer object_location
    if is_image_url(obj.object_location):
        return obj.object_location

    # Fallback to object_thumb
    if is_image_url(obj.object_thumb):
        return obj.object_thumb

    return None


def process_object(
    obj: MetadataObject,
    run_path: Path,
    ollama_config: OllamaConfig,
    image_config: ImageProcessingConfig,
    class_options: ClassificationOptions,
    top_k: int | None = None,
) -> tuple[list[str], ClassificationRecord | None]:
    """Process a single object through the pipeline.

    Args:
        obj: Metadata object to process
        run_path: Run directory path
        ollama_config: Ollama configuration
        image_config: Image processing configuration
        class_options: Classification options
        top_k: Maximum number of codes to extract

    Returns:
        Tuple of (codes, detailed_record)
    """
    objectid = obj.objectid

    # Select image URL
    image_url = select_image_url(obj)
    if not image_url:
        logger.warning(f"No image URL found for {objectid}")
        return [], None

    # Download image
    logger.info(f"Downloading image for {objectid} from {image_url}")
    raw_bytes = download_image(image_url)
    raw_sha256 = compute_sha256(raw_bytes)

    # Cache raw image
    cache_dir = run_path / "data" / "src"
    cache_image(raw_bytes, cache_dir, raw_sha256, "bin")

    # Process image
    logger.info(f"Processing image for {objectid}")
    processed_bytes = process_image(raw_bytes, image_config)
    processed_sha256 = compute_sha256(processed_bytes)

    # Save processed image
    processed_path = run_path / "data" / f"{objectid}.jpg"
    processed_path.write_bytes(processed_bytes)

    # Classify with Ollama
    logger.info(f"Classifying {objectid} with Ollama")
    response = classify_image(
        processed_bytes,
        ollama_config.model,
        ollama_config.url,
        class_options,
    )

    # Save classification artifacts
    save_classification_artifacts(
        objectid,
        {
            "model": ollama_config.model,
            "messages": [{"role": "user", "content": USER_PROMPT}],
            "options": class_options.model_dump(),
        },
        response,
        run_path / "classify",
    )

    # Extract codes
    raw_text = response.get("message", {}).get("content", "")
    codes = extract_codes(raw_text, top_k)

    logger.info(f"Extracted {len(codes)} codes for {objectid}: {codes}")

    # Build detailed record
    now_iso = datetime.now(UTC).isoformat()
    top_k_ranks = [
        IconclassCodeRank(code=code, rank=i + 1) for i, code in enumerate(codes)
    ]

    details = IconclassDetails(
        codes=codes,
        top_k=top_k_ranks,
        model=ollama_config.model,
        prompt=USER_PROMPT,
        temperature=class_options.temperature,
        num_ctx=class_options.num_ctx,
        num_predict=class_options.num_predict,
        raw_text=raw_text,
        image_sha256=processed_sha256,
        image_source=image_url,
        processed_image_path=f"data/{objectid}.jpg",
        timestamp=now_iso,
    )

    record = ClassificationRecord(
        objectid=objectid,
        subject=SubjectDetails(iconclass=details),
    )

    return codes, record


def run_pipeline(
    source_url: str,
    output_dir: Path,
    ollama_config: OllamaConfig,
    image_config: ImageProcessingConfig,
    class_options: ClassificationOptions,
    sampling_config: SamplingConfig,
    top_k: int | None = None,
) -> None:
    """Run the complete classification pipeline.

    Args:
        source_url: URL to metadata.json
        output_dir: Base output directory for runs
        ollama_config: Ollama configuration
        image_config: Image processing configuration
        class_options: Classification options
        sampling_config: Sampling configuration
        top_k: Maximum number of codes per object
    """
    # Create run directory
    run_path, run_id = create_run_directory(output_dir)

    # Setup logging
    setup_logging(run_path / "logs" / "pipeline.log")

    logger.info("Starting Iconclass classification pipeline")
    logger.info(f"Run ID: {run_id}")

    # Initialize manifest
    started = datetime.now(UTC).isoformat()
    manifest = RunManifest(
        run_id=run_id,
        source_url=source_url,
        git_commit=get_git_commit(),
        python=f"{sys.version_info.major}.{sys.version_info.minor}",
        platform=f"{platform.system()} {platform.machine()}",
        ollama=ollama_config,
        image_processing=image_config,
        options=class_options,
        sampling=sampling_config,
        counts=RunCounts(),
        started=started,
    )

    # Fetch and save metadata
    raw_metadata = fetch_metadata(source_url)
    (run_path / "raw" / "metadata.json").write_text(json.dumps(raw_metadata, indent=2))

    # Parse metadata
    metadata = Metadata(**raw_metadata)
    all_objects = metadata.objects

    manifest.counts.total = len(all_objects)
    logger.info(f"Total objects in metadata: {len(all_objects)}")

    # Filter to keep only children (m prefix), exclude parents (abb prefix)
    filtered_objects, abb_count = filter_children_only(all_objects)
    manifest.counts.abb_filtered = abb_count
    manifest.counts.m_included = len(filtered_objects)
    logger.info(
        f"After filtering: {abb_count} abb objects excluded, "
        f"{len(filtered_objects)} objects retained"
    )

    # Sample objects based on configuration
    sampled_objects = sample_objects(filtered_objects, sampling_config)
    manifest.counts.sampled = len(sampled_objects)
    logger.info(
        f"After sampling ({sampling_config.mode}): {len(sampled_objects)} objects"
    )

    # Fail-fast validation
    validate_no_abb_in_sample(sampled_objects)

    # Update metadata with filtered/sampled objects
    objects = sampled_objects

    # Process each object
    details_file = run_path / "results" / "iconclass_details.jsonl"

    for obj in tqdm(objects, desc="Classifying objects"):
        manifest.counts.attempted += 1

        try:
            codes, record = process_object(
                obj,
                run_path,
                ollama_config,
                image_config,
                class_options,
                top_k,
            )

            if codes:
                # Update object with codes
                obj.subject = codes
                manifest.counts.classified += 1

                # Save detailed record
                if record:
                    append_jsonl(details_file, record.model_dump())
            else:
                manifest.counts.skipped += 1

        except Exception as e:
            logger.error(f"Error processing {obj.objectid}: {e}", exc_info=True)
            manifest.counts.errors += 1

    # Save classified metadata while preserving extra fields
    # Use model_dump() with mode='json' to handle extra fields properly
    classified_metadata = metadata.model_dump(mode="json")
    classified_path = run_path / "results" / "metadata.classified.json"
    classified_path.write_text(json.dumps(classified_metadata, indent=2))

    # Finalize manifest
    manifest.finished = datetime.now(UTC).isoformat()
    save_manifest(run_path, manifest)

    logger.info("Pipeline completed")
    logger.info(f"Classified: {manifest.counts.classified}/{manifest.counts.total}")
    logger.info(f"Errors: {manifest.counts.errors}")
    logger.info(f"Results saved to: {run_path}")
