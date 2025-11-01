"""Unit tests for filtering and sampling."""

import pytest

from iconclass_classification.filtering import (
    filter_children_only,
    sample_objects,
    validate_no_abb_in_sample,
)
from iconclass_classification.models import MetadataObject, SamplingConfig


def test_filter_children_only():
    """Test filtering of abb (parent) objects."""
    objects = [
        MetadataObject(objectid="abb10039", object_location="http://example.com/1.jpg"),
        MetadataObject(objectid="m10040", object_location="http://example.com/2.jpg"),
        MetadataObject(objectid="abb10041", object_location="http://example.com/3.jpg"),
        MetadataObject(objectid="m10042", object_location="http://example.com/4.jpg"),
        MetadataObject(objectid="other123", object_location="http://example.com/5.jpg"),
    ]

    filtered, abb_count = filter_children_only(objects)

    assert abb_count == 2
    assert len(filtered) == 3
    assert all(not obj.objectid.lower().startswith("abb") for obj in filtered)
    assert filtered[0].objectid == "m10040"
    assert filtered[1].objectid == "m10042"
    assert filtered[2].objectid == "other123"


def test_filter_children_only_case_insensitive():
    """Test filtering is case-insensitive."""
    objects = [
        MetadataObject(objectid="ABB10039", object_location="http://example.com/1.jpg"),
        MetadataObject(objectid="M10040", object_location="http://example.com/2.jpg"),
        MetadataObject(objectid="Abb10041", object_location="http://example.com/3.jpg"),
    ]

    filtered, abb_count = filter_children_only(objects)

    assert abb_count == 2
    assert len(filtered) == 1
    assert filtered[0].objectid == "M10040"


def test_sample_full_mode():
    """Test full sampling mode returns all objects."""
    objects = [
        MetadataObject(objectid=f"m{i}", object_location=f"http://example.com/{i}.jpg")
        for i in range(10)
    ]

    config = SamplingConfig(mode="full")
    sampled = sample_objects(objects, config)

    assert len(sampled) == 10
    assert sampled == objects


def test_sample_random_mode():
    """Test random sampling mode."""
    objects = [
        MetadataObject(objectid=f"m{i}", object_location=f"http://example.com/{i}.jpg")
        for i in range(100)
    ]

    config = SamplingConfig(mode="random", size=10, seed=42)
    sampled = sample_objects(objects, config)

    assert len(sampled) == 10
    # Test reproducibility
    sampled2 = sample_objects(objects, config)
    assert [obj.objectid for obj in sampled] == [obj.objectid for obj in sampled2]


def test_sample_random_size_larger_than_total():
    """Test random sampling when size >= total returns all."""
    objects = [
        MetadataObject(objectid=f"m{i}", object_location=f"http://example.com/{i}.jpg")
        for i in range(5)
    ]

    config = SamplingConfig(mode="random", size=10, seed=42)
    sampled = sample_objects(objects, config)

    assert len(sampled) == 5


def test_validate_no_abb_success():
    """Test validation passes with no abb objects."""
    objects = [
        MetadataObject(objectid="m10040", object_location="http://example.com/2.jpg"),
        MetadataObject(objectid="m10042", object_location="http://example.com/4.jpg"),
    ]

    # Should not raise
    validate_no_abb_in_sample(objects)


def test_validate_no_abb_failure():
    """Test validation fails with abb objects."""
    objects = [
        MetadataObject(objectid="abb10039", object_location="http://example.com/1.jpg"),
        MetadataObject(objectid="m10040", object_location="http://example.com/2.jpg"),
    ]

    with pytest.raises(ValueError, match="abb \\(parent\\) objects"):
        validate_no_abb_in_sample(objects)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
