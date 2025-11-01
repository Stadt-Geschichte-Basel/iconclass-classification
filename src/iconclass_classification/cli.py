"""Command-line interface for Iconclass classification pipeline."""

from pathlib import Path

import click

from iconclass_classification.models import (
    ClassificationOptions,
    ImageProcessingConfig,
    OllamaConfig,
    SamplingConfig,
)
from iconclass_classification.pipeline import run_pipeline


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Iconclass classification pipeline using Ollama VLM."""
    pass


@cli.command()
@click.option(
    "--source",
    type=str,
    required=True,
    help="URL to metadata.json",
)
@click.option(
    "--model",
    type=str,
    default="hf.co/mradermacher/iconclass-vlm-GGUF:Q4_K_M",
    help="Ollama model name",
)
@click.option(
    "--ollama-url",
    type=str,
    default="http://localhost:11434",
    help="Ollama service URL",
)
@click.option(
    "--max-side",
    type=int,
    default=1024,
    help="Maximum image side length in pixels",
)
@click.option(
    "--quality",
    type=int,
    default=92,
    help="JPEG quality (1-100)",
)
@click.option(
    "--top-k",
    type=int,
    default=None,
    help="Maximum number of codes to extract per image",
)
@click.option(
    "--sampling-mode",
    type=click.Choice(["random", "fixed", "full"]),
    default="full",
    help="Sampling mode: random, fixed IDs, or full dataset",
)
@click.option(
    "--sampling-size",
    type=int,
    default=None,
    help="Number of objects to sample (for random mode)",
)
@click.option(
    "--sampling-seed",
    type=int,
    default=42,
    help="Random seed for reproducible sampling",
)
@click.option(
    "--fixed-ids-file",
    type=str,
    default=None,
    help="Path to file with fixed object IDs (for fixed mode)",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="runs",
    help="Base output directory for runs",
)
@click.option(
    "--prompt-template",
    type=click.Choice(["default", "instruction", "few_shot"]),
    default="default",
    help="Prompt template: default, instruction-based, or few-shot examples",
)
@click.option(
    "--temperature",
    type=float,
    default=0.0,
    help="Model temperature",
)
@click.option(
    "--num-ctx",
    type=int,
    default=4096,
    help="Context window size",
)
@click.option(
    "--num-predict",
    type=int,
    default=128,
    help="Maximum tokens to predict",
)
def classify(
    source: str,
    model: str,
    ollama_url: str,
    max_side: int,
    quality: int,
    top_k: int | None,
    sampling_mode: str,
    sampling_size: int | None,
    sampling_seed: int,
    fixed_ids_file: str | None,
    output: Path,
    prompt_template: str,
    temperature: float,
    num_ctx: int,
    num_predict: int,
):
    """Classify images using Ollama Iconclass VLM.

    Only processes children objects (m prefix). Parent objects (abb prefix) are
    automatically filtered out.

    Example:
        python -m iconclass_classification.cli classify \\
            --source https://forschung.stadtgeschichtebasel.ch/assets/data/metadata.json \\
            --sampling-mode random --sampling-size 10
    """
    # Build configurations
    ollama_config = OllamaConfig(url=ollama_url, model=model)
    image_config = ImageProcessingConfig(max_side=max_side, quality=quality)
    class_options = ClassificationOptions(
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
    )
    sampling_config = SamplingConfig(
        mode=sampling_mode,
        size=sampling_size,
        seed=sampling_seed,
        fixed_ids_file=fixed_ids_file,
    )

    # Run pipeline
    run_pipeline(
        source_url=source,
        output_dir=output,
        ollama_config=ollama_config,
        image_config=image_config,
        class_options=class_options,
        sampling_config=sampling_config,
        prompt_template=prompt_template,
        top_k=top_k,
    )

    click.echo(f"\nPipeline completed. Results saved to: {output}")


if __name__ == "__main__":
    cli()
