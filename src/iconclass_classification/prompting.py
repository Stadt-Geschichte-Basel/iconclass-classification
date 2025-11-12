"""Prompt templates and handling for Iconclass classification."""

import logging

logger = logging.getLogger(__name__)

# Default prompts
SYSTEM_PROMPT_DEFAULT = (
    "You are an Iconclass classifier. Given an artwork image, "
    "output only newline-separated Iconclass codes."
)

USER_PROMPT_DEFAULT = "Generate Iconclass labels for this image"

# Instruction-based prompt template
SYSTEM_PROMPT_INSTRUCTION = """You are an expert art historian specialized in the Iconclass classification system.

Your task is to analyze the given artwork image and identify appropriate Iconclass codes that describe its content.

Instructions:
1. Carefully observe all elements in the image
2. Identify subjects, themes, symbols, and iconographic elements
3. Output ONLY valid Iconclass codes, one per line
4. If you cannot identify any Iconclass codes, output: NONE

Example output format:
71H7131
25F2
31A2351"""

USER_PROMPT_INSTRUCTION = "Analyze this artwork image and provide Iconclass codes:"

# Few-shot prompt with examples
SYSTEM_PROMPT_FEW_SHOT = """You are an expert art historian specialized in the Iconclass classification system.

Here are examples of how to classify images:

Example 1: Religious scene with Mary and infant Jesus
Output:
11F4
71H7131

Example 2: Landscape with trees and river
Output:
25H17
25H21

Example 3: Portrait of a noble person
Output:
31A2351

Now analyze the following artwork image and provide appropriate Iconclass codes.
Output ONLY the codes, one per line. If no codes apply, output: NONE"""

USER_PROMPT_FEW_SHOT = "Classify this artwork:"

# Prompt templates registry
PROMPT_TEMPLATES = {
    "default": (SYSTEM_PROMPT_DEFAULT, USER_PROMPT_DEFAULT),
    "instruction": (SYSTEM_PROMPT_INSTRUCTION, USER_PROMPT_INSTRUCTION),
    "few_shot": (SYSTEM_PROMPT_FEW_SHOT, USER_PROMPT_FEW_SHOT),
}


def get_prompts(template: str = "default") -> tuple[str, str]:
    """Get system and user prompts for a given template.

    Args:
        template: Prompt template name (default, instruction, few_shot)

    Returns:
        Tuple of (system_prompt, user_prompt)

    Raises:
        ValueError: If template not found
    """
    if template not in PROMPT_TEMPLATES:
        raise ValueError(
            f"Unknown prompt template: {template}. "
            f"Available: {list(PROMPT_TEMPLATES.keys())}"
        )

    return PROMPT_TEMPLATES[template]


def log_empty_response_debug(
    objectid: str,
    raw_response: str,
    prompt_template: str,
    image_sha256: str,
) -> None:
    """Log debug information for empty classification responses.

    Args:
        objectid: Object identifier
        raw_response: Raw model response text
        prompt_template: Prompt template used
        image_sha256: SHA256 hash of processed image
    """
    logger.warning(
        f"Empty classification for {objectid}: model returned no valid codes"
    )
    logger.debug(f"  Object ID: {objectid}")
    logger.debug(f"  Prompt template: {prompt_template}")
    logger.debug(f"  Image SHA256: {image_sha256}")
    logger.debug(f"  Raw response length: {len(raw_response)} chars")
    logger.debug(f"  Raw response preview: {raw_response[:200]!r}")

    # Check if model explicitly said NONE
    if "NONE" in raw_response.upper():
        logger.info(f"  Model explicitly indicated no codes for {objectid}")
    elif not raw_response.strip():
        logger.warning(f"  Model returned completely empty response for {objectid}")
    else:
        logger.warning(
            f"  Model returned non-empty response but no codes extracted for {objectid}"
        )
