"""Pydantic models for the Iconclass classification pipeline."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class IconclassCodeRank(BaseModel):
    """Individual Iconclass code with rank."""

    code: str
    rank: int


class IconclassDetails(BaseModel):
    """Detailed Iconclass classification metadata."""

    codes: list[str]
    top_k: list[IconclassCodeRank]
    model: str
    prompt: str
    temperature: float
    num_ctx: int
    num_predict: int
    raw_text: str
    image_sha256: str
    image_source: str
    processed_image_path: str
    timestamp: str


class SubjectDetails(BaseModel):
    """Subject details wrapper for classification."""

    iconclass: IconclassDetails


class ClassificationRecord(BaseModel):
    """Complete classification record for JSONL output."""

    objectid: str
    subject: SubjectDetails


class ImageProcessingConfig(BaseModel):
    """Configuration for image processing."""

    max_side: int = 1024
    format: Literal["JPEG", "PNG", "WebP"] = "JPEG"
    colorspace: Literal["RGB", "RGBA", "L"] = "RGB"
    quality: int = 92


class OllamaConfig(BaseModel):
    """Ollama service configuration."""

    url: str = "http://localhost:11434"
    model: str = "hf.co/mradermacher/iconclass-vlm-GGUF:Q4_K_M"


class OpenRouterConfig(BaseModel):
    """OpenRouter service configuration."""

    api_key: SecretStr
    model: str = "qwen/qwen3-vl-235b-a22b-instruct"
    api_url: str = "https://openrouter.ai/api/v1/chat/completions"
    max_image_size: int = 2048  # Max dimension for image resize


class ClassificationOptions(BaseModel):
    """Classification options."""

    temperature: float = 0.0
    num_ctx: int = 4096
    num_predict: int = 128


class SamplingConfig(BaseModel):
    """Configuration for data sampling."""

    mode: Literal["random", "fixed", "full"] = "full"
    size: int | None = None
    seed: int = 42
    fixed_ids_file: str | None = None


class RunCounts(BaseModel):
    """Counts for a pipeline run."""

    total: int = 0
    abb_filtered: int = 0
    m_included: int = 0
    sampled: int = 0
    attempted: int = 0
    classified: int = 0
    skipped: int = 0
    errors: int = 0


class RunManifest(BaseModel):
    """Manifest for a pipeline run."""

    run_id: str
    source_url: str
    git_commit: str | None = None
    python: str
    platform: str
    backend: Literal["ollama", "openrouter"]
    ollama: OllamaConfig | None = None
    openrouter: OpenRouterConfig | None = None
    image_processing: ImageProcessingConfig
    options: ClassificationOptions
    sampling: SamplingConfig
    counts: RunCounts
    started: str
    finished: str | None = None


class MetadataObject(BaseModel):
    """A single object from metadata.json."""

    model_config = ConfigDict(extra="allow")

    objectid: str
    object_location: str | None = None
    object_thumb: str | None = None
    subject: list[str] = Field(default_factory=list)


class Metadata(BaseModel):
    """Root metadata structure."""

    model_config = ConfigDict(extra="allow")

    objects: list[MetadataObject]
