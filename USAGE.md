# Iconclass Classification Pipeline

A reproducible pipeline to classify digital objects using the Iconclass VLM (Vision-Language Model) running under Ollama on local infrastructure.

## Overview

This project implements an automated pipeline for classifying artwork images with [Iconclass](https://iconclass.org/) codes using a locally-hosted Ollama model. The pipeline downloads images, processes them, classifies them with the Iconclass VLM, and writes the results back to structured metadata files with full provenance tracking.

## Features

- 🎨 **Iconclass Classification**: Automated classification using the Iconclass VLM model via Ollama
- 📦 **Batch Processing**: Process entire collections from metadata.json files
- 🔄 **Image Processing**: Automatic download, resize, and normalization of images
- 💾 **Smart Caching**: SHA256-based deduplication of downloaded images
- 📊 **Dual Output**: Compact codes in metadata + detailed classification records
- 🔍 **Full Provenance**: Timestamped runs with complete audit trail
- 🛡️ **Robust**: Retry logic, error handling, and comprehensive logging
- 📐 **Type-Safe**: Built with Pydantic models for data validation

## Installation

### Prerequisites

- Python ≥ 3.11
- [Ollama](https://ollama.ai/) running locally
- The Iconclass VLM model pulled in Ollama:
  ```bash
  ollama pull hf.co/mradermacher/iconclass-vlm-GGUF:Q4_K_M
  ```

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/Stadt-Geschichte-Basel/iconclass-classification.git
   cd iconclass-classification
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   ```

## Usage

### Basic Usage

Classify images from a metadata.json URL:

```bash
python -m iconclass_classification classify \
  --source https://forschung.stadtgeschichtebasel.ch/assets/data/metadata.json
```

### Test with Sample Data

Process only the first 3 objects for testing:

```bash
python -m iconclass_classification classify \
  --source https://forschung.stadtgeschichtebasel.ch/assets/data/metadata.json \
  --sample 3
```

### Advanced Options

```bash
python -m iconclass_classification classify \
  --source https://forschung.stadtgeschichtebasel.ch/assets/data/metadata.json \
  --model hf.co/mradermacher/iconclass-vlm-GGUF:Q4_K_M \
  --ollama-url http://localhost:11434 \
  --max-side 1024 \
  --quality 92 \
  --top-k 5 \
  --output runs \
  --temperature 0.0 \
  --num-ctx 4096 \
  --num-predict 128
```

### Command-Line Options

| Option          | Default                                        | Description                         |
| --------------- | ---------------------------------------------- | ----------------------------------- |
| `--source`      | _required_                                     | URL to metadata.json                |
| `--model`       | `hf.co/mradermacher/iconclass-vlm-GGUF:Q4_K_M` | Ollama model name                   |
| `--ollama-url`  | `http://localhost:11434`                       | Ollama service URL                  |
| `--max-side`    | `1024`                                         | Maximum image side length in pixels |
| `--quality`     | `92`                                           | JPEG quality (1-100)                |
| `--top-k`       | `None`                                         | Maximum number of codes per image   |
| `--sample`      | `None`                                         | Process only first N objects        |
| `--output`      | `runs`                                         | Base output directory               |
| `--temperature` | `0.0`                                          | Model temperature                   |
| `--num-ctx`     | `4096`                                         | Context window size                 |
| `--num-predict` | `128`                                          | Maximum tokens to predict           |

## Output Structure

Each run creates a timestamped directory with the following structure:

```
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

### Running Tests

```bash
# Install dev dependencies
pip install pytest

# Run tests
pytest test/ -v
```

### Code Quality

```bash
# Format code
ruff format .

# Check code
ruff check .

# Auto-fix issues
ruff check --fix .
```

## Project Structure

```
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
python -m iconclass_classification classify \
  --source <URL> \
  --max-side 512 \
  --sample 10
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
