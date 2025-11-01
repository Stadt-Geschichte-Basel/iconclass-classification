"""Image download, processing, and caching utilities."""

import hashlib
import io
from pathlib import Path

import requests
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from iconclass_classification.models import ImageProcessingConfig


def compute_sha256(data: bytes) -> str:
    """Compute SHA256 hash of data.

    Args:
        data: Binary data to hash

    Returns:
        Hexadecimal SHA256 hash string
    """
    return hashlib.sha256(data).hexdigest()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def download_image(url: str, timeout: int = 30) -> bytes:
    """Download image from URL with retries.

    Args:
        url: Image URL
        timeout: Request timeout in seconds

    Returns:
        Raw image bytes

    Raises:
        requests.RequestException: If download fails after retries
    """
    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()
    return response.content


def is_image_url(url: str | None) -> bool:
    """Check if URL likely points to an image.

    Args:
        url: URL to check

    Returns:
        True if URL appears to be an image
    """
    if not url:
        return False

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }
    url_lower = url.lower()
    return any(url_lower.endswith(ext) for ext in image_extensions)


def process_image(
    src_bytes: bytes, config: ImageProcessingConfig | None = None
) -> bytes:
    """Process image: convert to RGB, resize, and compress.

    Args:
        src_bytes: Source image bytes
        config: Image processing configuration

    Returns:
        Processed image bytes in specified format

    Raises:
        PIL.UnidentifiedImageError: If image cannot be opened
    """
    if config is None:
        config = ImageProcessingConfig()

    # Open and convert to RGB
    img = Image.open(io.BytesIO(src_bytes)).convert(config.colorspace)

    # Resize if needed
    width, height = img.size
    max_side = max(width, height)
    if max_side > config.max_side:
        scale = config.max_side / max_side
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.LANCZOS)

    # Save to buffer
    buffer = io.BytesIO()
    img.save(buffer, config.format, quality=config.quality, optimize=True)
    return buffer.getvalue()


def cache_image(
    image_bytes: bytes, cache_dir: Path, sha256_hash: str, extension: str = "jpg"
) -> Path:
    """Cache image to disk using SHA256 hash as filename.

    Args:
        image_bytes: Image data to cache
        cache_dir: Directory for cached images
        sha256_hash: SHA256 hash for deduplication
        extension: File extension (without dot)

    Returns:
        Path to cached image file
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{sha256_hash}.{extension}"

    # Only write if not already cached
    if not cache_path.exists():
        cache_path.write_bytes(image_bytes)

    return cache_path
