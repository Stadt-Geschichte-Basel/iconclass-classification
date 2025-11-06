# Iconclass Classification Pipeline

**A reproducible pipeline to classify digital heritage objects using Iconclass VLM via Ollama.**

This project implements an automated, locally-hosted pipeline for classifying artwork images with [Iconclass](https://iconclass.org/) codes using a Vision-Language Model (VLM) running under [Ollama](https://ollama.ai/). Designed for the Stadt Geschichte Basel digital collections, the pipeline processes metadata, downloads images, classifies them, and writes results back with full provenance tracking.

[![GitHub issues](https://img.shields.io/github/issues/Stadt-Geschichte-Basel/iconclass-classification.svg)](https://github.com/Stadt-Geschichte-Basel/iconclass-classification/issues)
[![Code license](https://img.shields.io/github/license/Stadt-Geschichte-Basel/iconclass-classification.svg)](https://github.com/Stadt-Geschichte-Basel/iconclass-classification/blob/main/LICENSE-AGPL.md)
[![Data license](https://img.shields.io/badge/license-CC%20BY%204.0-blue.svg)](https://github.com/Stadt-Geschichte-Basel/iconclass-classification/blob/main/LICENSE-CCBY.md)

## Features

- 🎨 **Automated Iconclass Classification**: Uses the Iconclass VLM model via Ollama for accurate classification
- 📦 **Batch Processing**: Process entire collections from metadata.json files
- 🔄 **Smart Image Processing**: Automatic download, resize, normalization, and SHA256-based caching
- 🎯 **Intelligent Filtering**: Automatically filters to process only children (m) objects, excluding parents (abb)
- 🎲 **Flexible Sampling**: Random, fixed, or full dataset sampling modes with reproducible seeds
- 📝 **Multiple Prompts**: Three prompt templates (default, instruction, few-shot) for optimal results
- 📊 **Dual Output**: Compact codes in metadata + detailed JSONL records for auditability
- 🔍 **Full Provenance**: Timestamped run directories with complete audit trail (requests, responses, manifests)
- 🛡️ **Robust**: Built-in retry logic, comprehensive error handling, and logging
- 📐 **Type-Safe**: Pydantic models ensure data validation and consistency
- 🧪 **Tested**: Unit and integration tests ensure reliability

## Quick Start

### Prerequisites

- Python ≥ 3.11
- [Ollama](https://ollama.ai/) running locally
- The Iconclass VLM model:
  ```bash
  ollama pull hf.co/mradermacher/iconclass-vlm-GGUF:Q4_K_M
  ```

### Installation

```bash
# Clone the repository
git clone https://github.com/Stadt-Geschichte-Basel/iconclass-classification.git
cd iconclass-classification

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install requests pillow tqdm pydantic tenacity click
```

### Basic Usage

**Using Ollama (local):**

```bash
python -m iconclass_classification classify \
  --source https://forschung.stadtgeschichtebasel.ch/assets/data/metadata.json \
  --sampling-mode random \
  --sampling-size 10
```

**Using OpenRouter (cloud):**

```bash
export OPENROUTER_API_KEY=your_key_here
python -m iconclass_classification classify-openrouter \
  --source https://forschung.stadtgeschichtebasel.ch/assets/data/metadata.json \
  --sampling-mode random \
  --sampling-size 10
```

**Important**: The pipeline automatically filters to only process children objects (m prefix). Parent objects (abb prefix) are excluded from classification.

The pipeline will:

1. Fetch metadata from the source URL
2. Filter out abb (parent) objects, keep only m (children) objects
3. Sample objects based on your configuration
4. Download and process images
5. Classify each image with the selected backend (Ollama or OpenRouter)
6. Write results to `runs/<timestamp>/`

For detailed usage instructions and advanced features, see:

- [USAGE.md](USAGE.md) - Complete usage guide for both backends
- [training-and-prompting.md](documentation/training-and-prompting.md) - Prompt templates and troubleshooting

## Output Structure

Each run creates a timestamped directory with complete provenance:

```
runs/<UTC-ISO8601>/
  raw/metadata.json              # Original metadata
  data/
    src/                         # Cached images (deduplicated by SHA256)
    <objectid>.jpg               # Processed images
  classify/
    <objectid>_request.json      # Classification requests
    <objectid>_response.json     # Model responses
  logs/pipeline.log              # Detailed logs
  results/
    metadata.classified.json     # Metadata with Iconclass codes
    iconclass_details.jsonl      # Detailed classification records
  manifest.json                  # Run metadata & counts
```

### Example Output

**Compact metadata** (`metadata.classified.json`):

```json
{
	"objectid": "abb10039",
	"title": "Die Löblich und wyt berümpt Stat Basel",
	"subject": ["71H7131", "25F2"]
}
```

**Detailed record** (`iconclass_details.jsonl`):

```json
{
	"objectid": "abb10039",
	"subject": {
		"iconclass": {
			"codes": ["71H7131", "25F2"],
			"model": "hf.co/mradermacher/iconclass-vlm-GGUF:Q4_K_M",
			"image_sha256": "...",
			"timestamp": "2025-11-01T10:00:00Z"
		}
	}
}
```

## Architecture

The pipeline is modular and consists of these components:

```
src/iconclass_classification/
  models.py         # Pydantic data models
  image_utils.py    # Image download & processing
  ollama_client.py  # Ollama API integration
  pipeline.py       # Pipeline orchestration
  cli.py            # Command-line interface
```

### How It Works

1. **Fetch Metadata**: Download metadata.json from source URL
2. **Image Selection**: Choose best available image URL (prefer `object_location`, fallback to `object_thumb`)
3. **Download & Cache**: Download images with SHA256-based deduplication
4. **Process**: Resize to max size, convert to RGB, compress as JPEG
5. **Classify**: Send processed images to Ollama Iconclass VLM
6. **Extract**: Parse Iconclass codes from model responses using regex
7. **Write**: Update metadata with codes, save detailed JSONL records
8. **Log**: Save all requests, responses, and run manifest

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest

# Run all tests
PYTHONPATH=src:$PYTHONPATH pytest test/ -v

# Run only unit tests
PYTHONPATH=src:$PYTHONPATH pytest test/unit/ -v

# Run only integration tests
PYTHONPATH=src:$PYTHONPATH pytest test/integration/ -v
```

### Code Quality

```bash
# Format Python code
ruff format .

# Check Python code
ruff check .

# Auto-fix issues
ruff check --fix .

# Format all files
npm run format

# Check all files
npm run check
```

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

Reduce image size or use sample mode:

```bash
python -m iconclass_classification classify \
  --source <URL> \
  --max-side 512 \
  --sample 10
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Support

- 🐛 [Report a bug](https://github.com/Stadt-Geschichte-Basel/iconclass-classification/issues)
- 💬 [Ask a question](https://github.com/Stadt-Geschichte-Basel/iconclass-classification/discussions)
- 📚 [View documentation](USAGE.md)

## License

- **Code**: [AGPL-3.0](LICENSE-AGPL.md)
- **Data**: [CC BY 4.0](LICENSE-CCBY.md)

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

## Acknowledgments

This project builds upon:

- [Iconclass](https://iconclass.org/) classification system
- [Ollama](https://ollama.ai/) for local LLM hosting
- The [open-research-data-template](https://github.com/maehr/open-research-data-template) for infrastructure
- [Pydantic](https://pydantic.dev/) for data validation
- [Pillow](https://pillow.readthedocs.io/) for image processing

## Authors

- **Stadt Geschichte Basel** - [Website](https://www.stadtgeschichtebasel.ch/)

See also the list of [contributors](https://github.com/Stadt-Geschichte-Basel/iconclass-classification/graphs/contributors) who participated in this project.
