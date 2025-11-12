"""Unit tests for image utilities."""

import io

import pytest
from PIL import Image

from iconclass_classification.image_utils import (
    compute_sha256,
    is_image_url,
    process_image,
)
from iconclass_classification.models import ImageProcessingConfig


def test_compute_sha256():
    """Test SHA256 hash computation."""
    data = b"test data"
    hash1 = compute_sha256(data)
    hash2 = compute_sha256(data)

    # Should be deterministic
    assert hash1 == hash2
    # Should be 64 hex characters
    assert len(hash1) == 64
    assert all(c in "0123456789abcdef" for c in hash1)


def test_is_image_url():
    """Test image URL detection."""
    # Valid image URLs
    assert is_image_url("https://example.com/image.jpg")
    assert is_image_url("https://example.com/image.jpeg")
    assert is_image_url("https://example.com/image.png")
    assert is_image_url("https://example.com/image.gif")
    assert is_image_url("https://example.com/path/to/IMAGE.JPG")

    # Invalid URLs
    assert not is_image_url("https://example.com/file.pdf")
    assert not is_image_url("https://example.com/")
    assert not is_image_url(None)
    assert not is_image_url("")


def test_process_image():
    """Test image processing."""
    # Create a test image
    img = Image.new("RGB", (2000, 1500), color="red")
    buffer = io.BytesIO()
    img.save(buffer, "PNG")
    src_bytes = buffer.getvalue()

    # Process with default config
    config = ImageProcessingConfig(max_side=1024)
    processed = process_image(src_bytes, config)

    # Verify processed image
    processed_img = Image.open(io.BytesIO(processed))
    assert processed_img.format == "JPEG"
    assert processed_img.mode == "RGB"

    # Should be resized
    assert max(processed_img.size) <= 1024
    # Aspect ratio should be preserved
    original_ratio = 2000 / 1500
    processed_ratio = processed_img.width / processed_img.height
    assert abs(original_ratio - processed_ratio) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
