"""OpenRouter API client for Iconclass classification using Qwen3-VL."""

import base64
import json
import logging
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from iconclass_classification.models import ClassificationOptions, OpenRouterConfig
from iconclass_classification.prompting import get_prompts

logger = logging.getLogger(__name__)


def encode_image_base64(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string for OpenRouter.

    Args:
        image_bytes: Image data

    Returns:
        Base64-encoded data URI string
    """
    b64_str = base64.b64encode(image_bytes).decode("utf-8")
    # OpenRouter expects data URI format
    return f"data:image/jpeg;base64,{b64_str}"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def classify_image(
    image_bytes: bytes,
    config: OpenRouterConfig,
    options: ClassificationOptions | None = None,
    prompt_template: str = "default",
    timeout: int = 120,
) -> dict[str, Any]:
    """Classify image using OpenRouter API with Qwen3-VL.

    Args:
        image_bytes: Processed image bytes
        config: OpenRouter configuration
        options: Classification options
        prompt_template: Prompt template to use
        timeout: Request timeout in seconds

    Returns:
        Dictionary with raw response

    Raises:
        requests.RequestException: If classification fails after retries
    """
    if options is None:
        options = ClassificationOptions()

    # Get prompts for the template
    system_prompt, user_prompt = get_prompts(prompt_template)

    # Encode image to data URI
    image_data_uri = encode_image_base64(image_bytes)

    # Build OpenRouter request payload following their multimodal format
    # Reference: https://openrouter.ai/docs#vision
    # Use input_text/input_image types which are broadly supported across models.
    # Allow backward compatibility if user passes the older model ID variant
    model_id = config.model.replace("qwen/qwen-3-vl", "qwen/qwen3-vl")
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_image", "image_url": image_data_uri},
                ],
            },
        ],
        "temperature": options.temperature,
        "max_tokens": options.num_predict,
    }

    # Set headers with API key
    headers = {
        "Authorization": f"Bearer {config.api_key.get_secret_value()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Stadt-Geschichte-Basel/iconclass-classification",
        "X-Title": "Iconclass Classification Pipeline",
    }

    # Make request
    logger.debug(f"Sending request to OpenRouter: {config.model}")
    response = requests.post(
        config.api_url,
        json=payload,
        headers=headers,
        timeout=timeout,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError:
        # Log detailed error body to aid debugging (e.g., schema validation errors)
        try:
            err_body = response.json()
        except Exception:
            err_body = response.text
        logger.error(
            "OpenRouter request failed (%s): %s", response.status_code, err_body
        )
        raise

    result = response.json()

    # Normalize response to match Ollama format for compatibility
    # OpenRouter returns: {"choices": [{"message": {"content": "..."}}]}
    # We want: {"message": {"content": "..."}}
    if "choices" in result and len(result["choices"]) > 0:
        normalized = {
            "message": result["choices"][0]["message"],
            "model": result.get("model", config.model),
            "created_at": result.get("created"),
            "_prompt_template": prompt_template,
            "_usage": result.get("usage", {}),
        }
        return normalized

    # Fallback if format is unexpected
    logger.warning("Unexpected OpenRouter response format")
    result["_prompt_template"] = prompt_template
    return result


def save_classification_artifacts(
    objectid: str,
    request_payload: dict[str, Any],
    response: dict[str, Any],
    classify_dir: Any,
) -> None:
    """Save classification request and response as debug artifacts.

    Args:
        objectid: Object identifier
        request_payload: Classification request (base64 images truncated)
        response: Classification response
        classify_dir: Directory for classification artifacts
    """
    classify_dir.mkdir(parents=True, exist_ok=True)

    # Save request (truncate base64 images to save space)
    import copy

    request_copy = copy.deepcopy(request_payload)
    if "messages" in request_copy:
        for message in request_copy["messages"]:
            if "content" in message and isinstance(message["content"], list):
                for item in message["content"]:
                    item_type = item.get("type")
                    # Handle image_url type with dict shape {"image_url": {"url": ...}}
                    if (
                        item_type == "image_url"
                        and "image_url" in item
                        and "url" in item["image_url"]
                    ):
                        url = item["image_url"]["url"]
                        if url.startswith("data:"):
                            item["image_url"]["url"] = url[:50] + "...[truncated]"
                    # Handle input_image type which can be string or dict
                    elif item_type == "input_image" and "image_url" in item:
                        image_field = item["image_url"]
                        # If it's a string data URI, truncate directly
                        if isinstance(image_field, str) and image_field.startswith(
                            "data:"
                        ):
                            item["image_url"] = image_field[:50] + "...[truncated]"
                        # If it's a dict with a url key, truncate that
                        elif isinstance(image_field, dict) and "url" in image_field:
                            url = image_field["url"]
                            if url.startswith("data:"):
                                image_field["url"] = url[:50] + "...[truncated]"

    request_path = classify_dir / f"{objectid}_request.json"
    request_path.write_text(json.dumps(request_copy, indent=2))

    # Save response
    response_path = classify_dir / f"{objectid}_response.json"
    response_path.write_text(json.dumps(response, indent=2))
