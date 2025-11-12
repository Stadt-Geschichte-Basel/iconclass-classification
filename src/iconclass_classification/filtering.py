"""Data filtering and sampling utilities."""

import json
import logging
import random
from pathlib import Path

from iconclass_classification.models import MetadataObject, SamplingConfig

logger = logging.getLogger(__name__)


def filter_children_only(
    objects: list[MetadataObject],
) -> tuple[list[MetadataObject], int]:
    """Filter to keep only children (m prefix), exclude parents (abb prefix).

    Args:
        objects: List of metadata objects

    Returns:
        Tuple of (filtered objects, count of filtered abb objects)
    """
    filtered = []
    abb_count = 0

    for obj in objects:
        objectid = obj.objectid.lower()
        if objectid.startswith("abb"):
            abb_count += 1
            logger.debug(f"Filtering out parent object: {obj.objectid}")
        elif objectid.startswith("m"):
            filtered.append(obj)
        else:
            # Keep objects that don't start with abb or m
            filtered.append(obj)
            logger.debug(f"Including object with non-standard prefix: {obj.objectid}")

    logger.info(f"Filtered {abb_count} abb objects, kept {len(filtered)} objects")
    return filtered, abb_count


def load_fixed_ids(file_path: str) -> list[str]:
    """Load fixed object IDs from a file.

    Args:
        file_path: Path to file containing object IDs (one per line or JSON array)

    Returns:
        List of object IDs
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Fixed IDs file not found: {file_path}")

    content = path.read_text().strip()

    # Try JSON format first
    if content.startswith("["):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

    # Fall back to line-by-line
    return [line.strip() for line in content.split("\n") if line.strip()]


def sample_objects(
    objects: list[MetadataObject],
    config: SamplingConfig,
) -> list[MetadataObject]:
    """Sample objects based on configuration.

    Args:
        objects: List of metadata objects (already filtered)
        config: Sampling configuration

    Returns:
        Sampled list of objects
    """
    if config.mode == "full":
        logger.info(f"Using full dataset: {len(objects)} objects")
        return objects

    if config.mode == "random":
        if config.size is None or config.size >= len(objects):
            logger.info(f"Sample size >= total, using all {len(objects)} objects")
            return objects

        # Set seed for reproducibility
        random.seed(config.seed)
        sampled = random.sample(objects, config.size)
        logger.info(
            f"Random sampling: selected {len(sampled)}/{len(objects)} objects (seed={config.seed})"
        )
        return sampled

    if config.mode == "fixed":
        if config.fixed_ids_file is None:
            raise ValueError("fixed_ids_file must be provided for fixed sampling mode")

        fixed_ids = load_fixed_ids(config.fixed_ids_file)
        logger.info(f"Loaded {len(fixed_ids)} fixed IDs from {config.fixed_ids_file}")

        # Create lookup for efficiency
        id_set = set(fixed_ids)
        sampled = [obj for obj in objects if obj.objectid in id_set]

        # Check for abb objects in fixed IDs (fail-fast)
        abb_in_sample = [
            obj.objectid for obj in sampled if obj.objectid.lower().startswith("abb")
        ]
        if abb_in_sample:
            logger.error(
                f"Found {len(abb_in_sample)} abb objects in fixed sample: {abb_in_sample[:5]}"
            )
            raise ValueError(
                f"Fixed IDs contain {len(abb_in_sample)} abb (parent) objects. "
                "Only m (children) objects should be classified."
            )

        logger.info(f"Fixed sampling: selected {len(sampled)}/{len(objects)} objects")
        return sampled

    raise ValueError(f"Unknown sampling mode: {config.mode}")


def validate_no_abb_in_sample(objects: list[MetadataObject]) -> None:
    """Validate that no abb objects are in the sample (fail-fast).

    Args:
        objects: List of objects to validate

    Raises:
        ValueError: If any abb objects are found
    """
    abb_objects = [
        obj.objectid for obj in objects if obj.objectid.lower().startswith("abb")
    ]
    if abb_objects:
        raise ValueError(
            f"Found {len(abb_objects)} abb (parent) objects in sample: {abb_objects[:5]}. "
            "Only m (children) objects should be classified. "
            "This indicates a filtering bug."
        )
