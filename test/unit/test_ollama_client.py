"""Unit tests for Ollama client."""

import pytest

from iconclass_classification.ollama_client import extract_codes


def test_extract_codes():
    """Test Iconclass code extraction."""
    # Test with valid codes
    text = "71H7131\n25F2\n31A2351"
    codes = extract_codes(text)
    assert codes == ["71H7131", "25F2", "31A2351"]

    # Test with top_k limit
    codes = extract_codes(text, top_k=2)
    assert codes == ["71H7131", "25F2"]

    # Test with duplicates (should be deduplicated)
    text = "71H7131\n71H7131\n25F2"
    codes = extract_codes(text)
    assert codes == ["71H7131", "25F2"]

    # Test with mixed content
    text = "The image shows 71H7131 and also 25F2."
    codes = extract_codes(text)
    assert "71H7131" in codes
    assert "25F2" in codes

    # Test empty
    codes = extract_codes("")
    assert codes == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
