"""Ollama API client for Iconclass classification."""

import base64
import json
import re
from pathlib import Path
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from iconclass_classification.models import ClassificationOptions

SYSTEM_PROMPT = (
    "You are an Iconclass classifier. Given an artwork image, "
    "output only newline-separated Iconclass codes."
)

USER_PROMPT = "Generate Iconclass labels for this image"

# Pattern to match Iconclass codes (alphanumeric starting with digit or letter)
# Must have at least 2 characters, avoid single letters like "T"
CODE_PATTERN = re.compile(r"\b[0-9A-Z][0-9A-Z.\-]+\b")


def encode_image_base64(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string.

    Args:
        image_bytes: Image data

    Returns:
        Base64-encoded string
    """
    return base64.b64encode(image_bytes).decode("utf-8")


def extract_codes(text: str, top_k: int | None = None) -> list[str]:
    """Extract Iconclass codes from model response.

    Args:
        text: Raw model response text
        top_k: Maximum number of codes to return

    Returns:
        List of unique Iconclass codes in order of appearance
    """
    seen = set()
    codes = []

    for match in CODE_PATTERN.findall(text):
        if match not in seen:
            seen.add(match)
            codes.append(match)
            if top_k and len(codes) >= top_k:
                break

    return codes


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def classify_image(
    image_bytes: bytes,
    model: str,
    ollama_url: str,
    options: ClassificationOptions | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    """Classify image using Ollama API.

    Args:
        image_bytes: Processed image bytes
        model: Ollama model name
        ollama_url: Ollama service URL
        options: Classification options
        timeout: Request timeout in seconds

    Returns:
        Dictionary with raw response and extracted codes

    Raises:
        requests.RequestException: If classification fails after retries
    """
    if options is None:
        options = ClassificationOptions()

    # Encode image
    image_b64 = encode_image_base64(image_bytes)

    # Build request payload
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT,
                "images": [image_b64],
            },
        ],
        "stream": False,
        "options": {
            "temperature": options.temperature,
            "num_ctx": options.num_ctx,
            "num_predict": options.num_predict,
        },
    }

    # Make request
    response = requests.post(
        f"{ollama_url}/api/chat",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()

    return response.json()


def save_classification_artifacts(
    objectid: str,
    request_payload: dict[str, Any],
    response: dict[str, Any],
    classify_dir: Path,
) -> None:
    """Save classification request and response as debug artifacts.

    Args:
        objectid: Object identifier
        request_payload: Classification request (without base64 image)
        response: Classification response
        classify_dir: Directory for classification artifacts
    """
    classify_dir.mkdir(parents=True, exist_ok=True)

    # Save request (strip base64 image data to save space)
    request_copy = json.loads(json.dumps(request_payload))
    for message in request_copy.get("messages", []):
        if "images" in message:
            message["images"] = ["<base64-encoded-image>"]

    request_path = classify_dir / f"{objectid}_request.json"
    request_path.write_text(json.dumps(request_copy, indent=2))

    # Save response
    response_path = classify_dir / f"{objectid}_response.json"
    response_path.write_text(json.dumps(response, indent=2))
