# Iconclass Classification Pipeline

A reproducible pipeline to classify digital objects using Vision-Language Models for Iconclass classification. Supports both local (Ollama) and cloud-based (OpenRouter) backends.

## Overview

This project implements an automated pipeline for classifying artwork images with [Iconclass](https://iconclass.org/) codes. The pipeline downloads images, processes them, classifies them using VLMs, and writes the results back to structured metadata files with full provenance tracking.

## Features

- 🎨 **Multiple Backends**: Local (Ollama Iconclass VLM) or Cloud (OpenRouter Qwen3-VL)
- 📦 **Batch Processing**: Process entire collections from metadata.json files
- 🔄 **Image Processing**: Automatic download, resize, and normalization of images
- 💾 **Smart Caching**: SHA256-based deduplication of downloaded images
- 📊 **Dual Output**: Compact codes in metadata + detailed classification records
- 🔍 **Full Provenance**: Timestamped runs with complete audit trail
- 🛡️ **Robust**: Retry logic, error handling, and comprehensive logging
- 📐 **Type-Safe**: Built with Pydantic models for data validation

## Installation

### Prerequisites

**For Ollama backend:**

- Python ≥ 3.11
- [Ollama](https://ollama.ai/) running locally
- The Iconclass VLM model pulled in Ollama:

  ```bash
  ollama pull hf.co/mradermacher/iconclass-vlm-GGUF:Q4_K_M
  ```

**For OpenRouter backend:**

- Python ≥ 3.11
- OpenRouter API key from [openrouter.ai](https://openrouter.ai/)

### Setup (with uv)

1. Clone the repository:

   ```bash
   git clone https://github.com/Stadt-Geschichte-Basel/iconclass-classification.git
   cd iconclass-classification
   ```

2. Install uv if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: pip install uv
```

1. Sync dependencies (creates and populates `.venv`):

```bash
uv sync
# (Optional) include dev tools (ruff, ty, pytest)
uv sync --group dev
```

1. Verify CLI is available:

```bash
uv run iconclass-classification --help
```

## Usage

### Important: Data Filtering

**The pipeline automatically filters data to process only children objects:**

- ✅ **Processes**: Objects with `m` prefix (children/individual objects)
- ❌ **Excludes**: Objects with `abb` prefix (parents/aggregates)

This filtering happens automatically before sampling. You don't need to pre-filter your data.

### Basic Usage (OpenRouter)

Classify images from a metadata.json URL (processes all children objects):

```bash
uv run iconclass-classification classify \
  --source https://forschung.stadtgeschichtebasel.ch/assets/data/metadata.json
```

### Sampling Modes

#### Random Sampling (recommended for testing)

Process a random sample with a fixed seed for reproducibility:

```bash
uv run iconclass-classification classify \
  --source https://forschung.stadtgeschichtebasel.ch/assets/data/metadata.json \
  --sampling-mode random \
  --sampling-size 10 \
  --sampling-seed 42
```

#### Fixed Sampling (specific objects)

Process only specific objects listed in a file:

```bash
# Create file with object IDs (one per line)
echo "m10039" > my_objects.txt
echo "m10040" >> my_objects.txt

uv run iconclass-classification classify \
  --source https://forschung.stadtgeschichtebasel.ch/assets/data/metadata.json \
  --sampling-mode fixed \
  --fixed-ids-file my_objects.txt
```

#### Full Dataset

Process all children objects (default if no sampling specified):

```bash
uv run iconclass-classification classify \
  --source https://forschung.stadtgeschichtebasel.ch/assets/data/metadata.json \
  --sampling-mode full
```

### Prompt Templates

The pipeline includes three prompt templates optimized for different scenarios:

#### Default (fastest, good for testing)

```bash
uv run iconclass-classification classify \
  --source https://forschung.stadtgeschichtebasel.ch/assets/data/metadata.json \
  --prompt-template default \
  --sampling-mode random --sampling-size 10
```

#### Instruction (recommended for production)

More detailed instructions with explicit NONE fallback:

```bash
uv run iconclass-classification classify \
  --source https://forschung.stadtgeschichtebasel.ch/assets/data/metadata.json \
  --prompt-template instruction \
  --sampling-mode random --sampling-size 10
```

#### Few-Shot (best for complex images)

Includes example classifications to guide the model:

```bash
uv run iconclass-classification classify \
  --source https://forschung.stadtgeschichtebasel.ch/assets/data/metadata.json \
  --prompt-template few_shot \
  --sampling-mode random --sampling-size 10
```

## OpenRouter Backend (Cloud-based)

### Basic Usage

To use OpenRouter with Qwen3-VL, you need an API key:

```bash
# Set your API key as environment variable
export OPENROUTER_API_KEY=your_key_here

# Run classification
uv run iconclass-classification classify-openrouter \
  --source https://forschung.stadtgeschichtebasel.ch/assets/data/metadata.json \
  --sampling-mode random --sampling-size 10
```

### With Different Models

OpenRouter supports various models. You can specify a different model:

```bash
uv run iconclass-classification classify-openrouter \
  --source https://forschung.stadtgeschichtebasel.ch/assets/data/metadata.json \
  --model qwen/qwen-3-vl-235b-a22b-instruct \
  --sampling-mode random --sampling-size 10
```

### Prompt Templates with OpenRouter

Just like Ollama, you can use different prompt templates:

```bash
uv run iconclass-classification classify-openrouter \
  --source https://forschung.stadtgeschichtebasel.ch/assets/data/metadata.json \
  --prompt-template instruction \
  --sampling-mode random --sampling-size 10
```

### Advanced Options (Ollama)

```bash
uv run iconclass-classification classify \
  --source https://forschung.stadtgeschichtebasel.ch/assets/data/metadata.json \
  --model hf.co/mradermacher/iconclass-vlm-GGUF:Q4_K_M \
  --ollama-url http://localhost:11434 \
  --prompt-template instruction \
  --sampling-mode random \
  --sampling-size 100 \
  --sampling-seed 42 \
  --max-side 1024 \
  --quality 92 \
  --top-k 5 \
  --output runs \
  --temperature 0.0 \
  --num-ctx 4096 \
  --num-predict 128
```

### Command-Line Options

#### Ollama Backend (`classify`)

| Option              | Default                                        | Description                                      |
| ------------------- | ---------------------------------------------- | ------------------------------------------------ |
| `--source`          | _required_                                     | URL to metadata.json                             |
| `--model`           | `hf.co/mradermacher/iconclass-vlm-GGUF:Q4_K_M` | Ollama model name                                |
| `--ollama-url`      | `http://localhost:11434`                       | Ollama service URL                               |
| `--prompt-template` | `default`                                      | Prompt template (default, instruction, few_shot) |
| `--sampling-mode`   | `full`                                         | Sampling mode (random, fixed, full)              |
| `--sampling-size`   | `None`                                         | Number of objects to sample (random mode)        |
| `--sampling-seed`   | `42`                                           | Random seed for reproducibility                  |
| `--fixed-ids-file`  | `None`                                         | File with object IDs (fixed mode)                |
| `--max-side`        | `1024`                                         | Maximum image side length in pixels              |
| `--quality`         | `92`                                           | JPEG quality (1-100)                             |
| `--top-k`           | `None`                                         | Maximum number of codes per image                |
| `--output`          | `runs`                                         | Base output directory                            |
| `--temperature`     | `0.0`                                          | Model temperature                                |
| `--num-ctx`         | `4096`                                         | Context window size                              |
| `--num-predict`     | `128`                                          | Maximum tokens to predict                        |

#### OpenRouter Backend (`classify-openrouter`)

| Option              | Default                              | Description                                      |
| ------------------- | ------------------------------------ | ------------------------------------------------ |
| `--source`          | _required_                           | URL to metadata.json                             |
| `--api-key`         | _required_ (or `OPENROUTER_API_KEY`) | OpenRouter API key                               |
| `--model`           | `qwen/qwen-3-vl-235b-a22b-instruct`  | OpenRouter model name                            |
| `--prompt-template` | `default`                            | Prompt template (default, instruction, few_shot) |
| `--sampling-mode`   | `full`                               | Sampling mode (random, fixed, full)              |
| `--sampling-size`   | `None`                               | Number of objects to sample (random mode)        |
| `--sampling-seed`   | `42`                                 | Random seed for reproducibility                  |
| `--fixed-ids-file`  | `None`                               | File with object IDs (fixed mode)                |
| `--max-side`        | `2048`                               | Maximum image side length in pixels              |
| `--quality`         | `92`                                 | JPEG quality (1-100)                             |
| `--top-k`           | `None`                               | Maximum number of codes per image                |
| `--output`          | `runs`                               | Base output directory                            |
| `--temperature`     | `0.0`                                | Model temperature                                |
| `--num-predict`     | `128`                                | Maximum tokens to predict                        |

## Output Structure

Each run creates a timestamped directory with the following structure:

```text
runs/<UTC-ISO8601>/
  raw/
    metadata.json              # Original metadata
  data/
    src/                       # Cached raw images (by SHA256)
    <objectid>.jpg             # Processed images
  classify/
    <objectid>_request.json    # Classification requests
    <objectid>_response.json   # Classification responses
  logs/
    pipeline.log               # Detailed logs
  results/
    metadata.classified.json   # Metadata with subject codes
    iconclass_details.jsonl    # Detailed classification records
  manifest.json                # Run metadata
```

## Output Formats

### Compact Metadata (metadata.classified.json)

Each object gains a flat `subject` array with Iconclass codes:

```json
{
	"objectid": "abb10039",
	"title": "Die Löblich und wyt berümpt Stat Basel",
	"subject": ["71H7131", "25F2"]
}
```

### Detailed Classification (iconclass_details.jsonl)

One JSON record per line with complete metadata:

```json
{
	"objectid": "abb10039",
	"subject": {
		"iconclass": {
			"codes": ["71H7131", "25F2"],
			"top_k": [
				{ "code": "71H7131", "rank": 1 },
				{ "code": "25F2", "rank": 2 }
			],
			"model": "hf.co/mradermacher/iconclass-vlm-GGUF:Q4_K_M",
			"prompt": "Generate Iconclass labels for this image",
			"temperature": 0.0,
			"num_ctx": 4096,
			"num_predict": 128,
			"raw_text": "<model output>",
			"image_sha256": "<sha256>",
			"image_source": "<URL>",
			"processed_image_path": "data/abb10039.jpg",
			"timestamp": "2025-11-01T10:00:00Z"
		}
	}
}
```

## Development

### Running Tests (uv)

```bash
# Ensure dev dependencies installed
uv sync --group dev

# Run tests
uv run pytest test/ -v
```

### Code Quality

```bash
# Format code
uv run ruff format .

# Check code
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# (Optional) Type check
uv run ty check
```

## Project Structure

```text
src/iconclass_classification/
  __init__.py           # Package initialization
  __main__.py           # Entry point
  cli.py                # Command-line interface
  models.py             # Pydantic data models
  image_utils.py        # Image processing utilities
  ollama_client.py      # Ollama API client
  pipeline.py           # Main pipeline orchestration

test/
  unit/                 # Unit tests
    test_image_utils.py
```

## Architecture

The pipeline consists of several modular components:

1. **Data Models** (`models.py`): Pydantic models for type-safe data handling
2. **Image Processing** (`image_utils.py`): Download, resize, normalize, and cache images
3. **Ollama Client** (`ollama_client.py`): Interface with Ollama API for classification
4. **Pipeline** (`pipeline.py`): Orchestrates the entire workflow
5. **CLI** (`cli.py`): User-friendly command-line interface

## How It Works

1. **Fetch Metadata**: Download metadata.json from the source URL
2. **Select Images**: Choose best image URL (prefer object_location, fallback to object_thumb)
3. **Download & Cache**: Download images with SHA256-based deduplication
4. **Process Images**: Resize to max_side, convert to RGB, compress as JPEG
5. **Classify**: Send processed images to Ollama Iconclass VLM
6. **Extract Codes**: Parse Iconclass codes from model responses
7. **Write Results**: Update metadata with codes, save detailed records
8. **Log Everything**: Save requests, responses, and run manifest

## Security & Ethics

- ✅ Only HTTPS downloads supported
- ✅ No storage of base64-encoded images on disk
- ✅ Respects image licensing information
- ✅ Classifications treated as metadata (recommended CC0 publication)

## Performance

- **Caching**: Images deduplicated by SHA256 hash
- **Concurrency**: Sequential processing (parallel processing can be added)
- **Resumability**: Each run is independent; failed runs can be retried

## Troubleshooting

For detailed troubleshooting, see [training-and-prompting.md](documentation/training-and-prompting.md).

### Empty Classifications

If objects are returning no Iconclass codes:

1. **Check the logs**: `runs/<timestamp>/logs/pipeline.log` contains debug information
2. **Review responses**: Check `classify/<objectid>_response.json` for model output
3. **Try different prompts**: Use `--prompt-template instruction` or `few_shot`
4. **Inspect images**: Check `data/<objectid>.jpg` for quality issues

### Ollama Not Running

Ensure Ollama is running:

```bash
ollama serve
```

### Model Not Found

Pull the Iconclass VLM model:

```bash
ollama pull hf.co/mradermacher/iconclass-vlm-GGUF:Q4_K_M
```

### Memory Issues

Reduce image size or batch size:

```bash
uv run iconclass-classification classify \
  --source <URL> \
  --max-side 512 \
  --sampling-mode random --sampling-size 10
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

- Code: [AGPL-3.0](LICENSE-AGPL.md)
- Data: [CC BY 4.0](LICENSE-CCBY.md)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{iconclass_classification,
  title = {Iconclass Classification Pipeline},
  author = {Stadt Geschichte Basel},
  year = {2025},
  url = {https://github.com/Stadt-Geschichte-Basel/iconclass-classification}
}
```

## Support

For questions, issues, or contributions:

- 🐛 [Report a bug](https://github.com/Stadt-Geschichte-Basel/iconclass-classification/issues)
- 💬 [Ask a question](https://github.com/Stadt-Geschichte-Basel/iconclass-classification/discussions)
- 📖 [Read the documentation](https://github.com/Stadt-Geschichte-Basel/iconclass-classification)

## Acknowledgments

This project uses:

- [Ollama](https://ollama.ai/) for local LLM hosting
- [Iconclass](https://iconclass.org/) classification system
- [Pydantic](https://pydantic.dev/) for data validation
- [Pillow](https://pillow.readthedocs.io/) for image processing
