"""Command-line interface for Iconclass classification pipeline."""

from pathlib import Path

import click

from iconclass_classification.models import (
    ClassificationOptions,
    ImageProcessingConfig,
    OllamaConfig,
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
    "--sample",
    type=int,
    default=None,
    help="Process only first N objects (for testing)",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default="runs",
    help="Base output directory for runs",
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
    sample: int | None,
    output: Path,
    temperature: float,
    num_ctx: int,
    num_predict: int,
):
    """Classify images using Ollama Iconclass VLM.

    Example:
        python -m iconclass_classification.cli classify \\
            --source https://forschung.stadtgeschichtebasel.ch/assets/data/metadata.json \\
            --sample 3
    """
    # Build configurations
    ollama_config = OllamaConfig(url=ollama_url, model=model)
    image_config = ImageProcessingConfig(max_side=max_side, quality=quality)
    class_options = ClassificationOptions(
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
    )

    # Run pipeline
    run_pipeline(
        source_url=source,
        output_dir=output,
        ollama_config=ollama_config,
        image_config=image_config,
        class_options=class_options,
        top_k=top_k,
        sample=sample,
    )

    click.echo(f"\nPipeline completed. Results saved to: {output}")


if __name__ == "__main__":
    cli()
