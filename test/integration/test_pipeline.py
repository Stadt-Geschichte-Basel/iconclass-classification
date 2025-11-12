"""Integration tests for the complete pipeline."""

import json
from unittest.mock import patch

import pytest

from iconclass_classification.models import (
    ClassificationOptions,
    ImageProcessingConfig,
    OllamaConfig,
    SamplingConfig,
)
from iconclass_classification.pipeline import run_pipeline


@pytest.fixture
def mock_metadata():
    """Create mock metadata structure."""
    return {
        "objects": [
            {
                "objectid": "test001",
                "object_location": "https://example.com/image1.jpg",
                "title": "Test Image 1",
            },
            {
                "objectid": "test002",
                "object_location": "https://example.com/image2.jpg",
                "title": "Test Image 2",
            },
        ]
    }


@pytest.fixture
def mock_ollama_response():
    """Create mock Ollama response."""
    return {
        "message": {"content": "71H7131\n25F2"},
        "model": "test-model",
        "created_at": "2025-11-01T10:00:00Z",
    }


def test_pipeline_structure(tmp_path, mock_metadata, mock_ollama_response):
    """Test that pipeline creates correct directory structure."""
    # Create a proper small test image using PIL
    import io

    from PIL import Image

    test_img = Image.new("RGB", (10, 10), color="red")
    img_buffer = io.BytesIO()
    test_img.save(img_buffer, "PNG")
    test_image = img_buffer.getvalue()

    # Mock external calls
    with (
        patch("iconclass_classification.pipeline.fetch_metadata") as mock_fetch,
        patch("iconclass_classification.pipeline.download_image") as mock_download,
        patch("iconclass_classification.ollama_client.classify_image") as mock_classify,
    ):
        # Setup mocks
        mock_fetch.return_value = mock_metadata
        mock_download.return_value = test_image
        mock_classify.return_value = mock_ollama_response

        # Run pipeline
        output_dir = tmp_path / "test_runs"
        run_pipeline(
            source_url="https://example.com/metadata.json",
            output_dir=output_dir,
            backend="ollama",
            ollama_config=OllamaConfig(),
            openrouter_config=None,
            image_config=ImageProcessingConfig(max_side=512),
            class_options=ClassificationOptions(),
            sampling_config=SamplingConfig(mode="full"),
            prompt_template="default",
        )

        # Check that run directory was created
        run_dirs = list(output_dir.glob("*"))
        assert len(run_dirs) == 1
        run_dir = run_dirs[0]

        # Check expected directories exist
        assert (run_dir / "raw").exists()
        assert (run_dir / "data").exists()
        assert (run_dir / "classify").exists()
        assert (run_dir / "logs").exists()
        assert (run_dir / "results").exists()

        # Check expected files exist
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "raw" / "metadata.json").exists()
        assert (run_dir / "logs" / "pipeline.log").exists()
        assert (run_dir / "results" / "metadata.classified.json").exists()
        assert (run_dir / "results" / "iconclass_details.jsonl").exists()

        # Check manifest structure
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert "run_id" in manifest
        assert "source_url" in manifest
        assert "counts" in manifest
        assert manifest["counts"]["total"] == 2

        # Check classified metadata
        classified_path = run_dir / "results" / "metadata.classified.json"
        classified = json.loads(classified_path.read_text())
        assert "objects" in classified
        assert len(classified["objects"]) == 2

        # Each object should have subject codes
        for obj in classified["objects"]:
            assert "subject" in obj
            assert isinstance(obj["subject"], list)

        # Check details file
        details_path = run_dir / "results" / "iconclass_details.jsonl"
        details_lines = details_path.read_text().strip().split("\n")
        assert len(details_lines) == 2

        for line in details_lines:
            record = json.loads(line)
            assert "objectid" in record
            assert "subject" in record
            assert "iconclass" in record["subject"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
