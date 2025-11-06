# Training and Prompting Guide for Iconclass VLM

This document provides guidance on training, fine-tuning, and effective prompting for Iconclass Vision-Language Models across different backends.

## Supported Backends

### 1. Ollama (Local)

- **Iconclass VLM Model**: [small-models-for-glam/iconclass-vlm](https://huggingface.co/small-models-for-glam/iconclass-vlm)
  - Pre-trained vision-language model specialized for Iconclass classification
  - Based on fine-tuned VLM architecture
  - Optimized for GLAM (Galleries, Libraries, Archives, Museums) domain
  - **Blog Post**: [Fine-tuning VLMs for Iconclass with TRL](https://danielvanstrien.xyz/posts/2025/iconclass-vlm-sft/trl-vlm-fine-tuning-iconclass.html) by Daniel van Strien

### 2. OpenRouter (Cloud)

- **Qwen3-VL**: [Qwen3-VL-235B-A22B-Instruct](https://openrouter.ai/qwen/qwen-3-vl-235b-a22b-instruct)
  - State-of-the-art multimodal model for vision-language tasks
  - Accessible via OpenRouter API
  - Large context window and strong performance on classification tasks
  - **Official Repo**: [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
  - **OpenRouter Docs**: [openrouter.ai/docs](https://openrouter.ai/docs)

## Choosing a Backend

| Feature         | Ollama (Local)                | OpenRouter (Cloud)             |
| --------------- | ----------------------------- | ------------------------------ |
| **Cost**        | Free (local compute)          | Pay per API call               |
| **Privacy**     | Complete data privacy         | Data sent to cloud             |
| **Performance** | Depends on local hardware     | Consistent, high performance   |
| **Setup**       | Requires Ollama + model pull  | Just API key                   |
| **Best for**    | Large batches, sensitive data | Quick testing, smaller batches |

## Resources

## Data Filtering Best Practices

### Only Classify Children Objects

**Important**: When working with the Basel dataset or similar hierarchical data:

- **DO classify**: Objects with `m` prefix (children objects)
- **DO NOT classify**: Objects with `abb` prefix (parent/aggregate objects)

Parent objects (`abb`) represent aggregations or collections and should not be classified individually. The pipeline automatically filters these out.

### Sampling Modes

The pipeline supports three sampling modes:

1. **Full** (`--sampling-mode full`): Process all children objects
2. **Random** (`--sampling-mode random --sampling-size N --sampling-seed 42`): Random sample of N objects with fixed seed for reproducibility
3. **Fixed** (`--sampling-mode fixed --fixed-ids-file ids.txt`): Process specific objects listed in a file

Example:

```bash
python -m iconclass_classification classify \
  --source https://forschung.stadtgeschichtebasel.ch/assets/data/metadata.json \
  --sampling-mode random \
  --sampling-size 100 \
  --sampling-seed 42
```

## Prompt Engineering

### Available Prompt Templates

The pipeline includes three prompt templates optimized for different scenarios:

#### 1. Default (`--prompt-template default`)

Simple, direct instruction for the model.

**Best for**: Quick testing, general-purpose classification

#### 2. Instruction (`--prompt-template instruction`)

Detailed instructions with explicit guidance and NONE fallback.

**Best for**: Improved accuracy, handling edge cases

Features:

- Step-by-step instructions
- Explicit NONE output for unclear images
- More structured format

#### 3. Few-Shot (`--prompt-template few_shot`)

Includes example classifications to guide the model.

**Best for**: Complex images, improving consistency

Features:

- Example input-output pairs
- Demonstrates expected format
- May improve model understanding

### Usage Example

```bash
python -m iconclass_classification classify \
  --source https://forschung.stadtgeschichtebasel.ch/assets/data/metadata.json \
  --prompt-template instruction \
  --sampling-mode random \
  --sampling-size 50
```

## Troubleshooting Empty Outputs

### Common Causes

1. **Image Quality Issues**
   - Low resolution or heavily degraded images
   - Solution: Check `image_sha256` in details log, review source image

2. **Model Uncertainty**
   - Image content doesn't clearly match known Iconclass categories
   - Solution: Try different prompt templates, especially `instruction` or `few_shot`

3. **Processing Errors**
   - Image format or conversion issues
   - Solution: Check pipeline logs for warnings during image processing

### Debugging Empty Classifications

When the pipeline encounters empty classifications, it automatically logs debug information:

```
WARNING  Empty classification for m10039: model returned no valid codes
DEBUG    Object ID: m10039
DEBUG    Prompt template: default
DEBUG    Image SHA256: abc123...
DEBUG    Raw response length: 45 chars
DEBUG    Raw response preview: 'The image shows...'
```

### Steps to Debug

1. **Check the logs**: Look in `runs/<timestamp>/logs/pipeline.log`
2. **Review raw responses**: Check `classify/<objectid>_response.json`
3. **Inspect the image**: Find it in `data/<objectid>.jpg`
4. **Try different prompts**: Experiment with `instruction` or `few_shot` templates
5. **Adjust temperature**: Lower temperature (0.0) for consistency, higher (0.3-0.7) for creativity

### Experimenting with Prompts

To test different prompts on a fixed sample:

```bash
# Create a file with specific object IDs
echo "m10039" > test_ids.txt
echo "m10040" >> test_ids.txt

# Test default prompt
python -m iconclass_classification classify \
  --source https://forschung.stadtgeschichtebasel.ch/assets/data/metadata.json \
  --sampling-mode fixed \
  --fixed-ids-file test_ids.txt \
  --prompt-template default \
  --output runs/test-default

# Test instruction prompt
python -m iconclass_classification classify \
  --source https://forschung.stadtgeschichtebasel.ch/assets/data/metadata.json \
  --sampling-mode fixed \
  --fixed-ids-file test_ids.txt \
  --prompt-template instruction \
  --output runs/test-instruction

# Compare results
diff runs/test-default/*/results/iconclass_details.jsonl \
     runs/test-instruction/*/results/iconclass_details.jsonl
```

## Model Parameters

### Temperature

Controls randomness in model outputs:

- `0.0`: Deterministic, consistent results (recommended for production)
- `0.3-0.5`: Slight variation, may improve recall
- `0.7-1.0`: More creative, less consistent

### Context Window (num_ctx)

- Default: 4096 tokens
- Increase if using very detailed prompts or few-shot examples
- Decrease to speed up inference

### Prediction Length (num_predict)

- Default: 128 tokens
- Usually sufficient for 5-10 Iconclass codes
- Increase if expecting many codes per image

## Best Practices

### For Production Use

1. **Use full dataset**: `--sampling-mode full`
2. **Fixed temperature**: `--temperature 0.0`
3. **Consistent prompt**: Stick with one template after testing
4. **Log everything**: Keep all artifacts for auditability

### For Experimentation

1. **Small samples**: `--sampling-mode random --sampling-size 10-100`
2. **Fixed seed**: Always use same seed for reproducibility
3. **Try all templates**: Compare results across prompts
4. **Adjust parameters**: Test different temperature and context settings

### For Debugging

1. **Enable debug logging**: Check logs in `runs/*/logs/pipeline.log`
2. **Review artifacts**: Inspect `classify/*_response.json` files
3. **Visual inspection**: Check processed images in `data/*.jpg`
4. **Fixed samples**: Use `--sampling-mode fixed` with problematic objects

## Performance Optimization

### Speed

- Use smaller context windows if possible
- Process in batches during off-peak hours
- Consider parallel processing (future enhancement)

### Quality

- Start with `instruction` template for best accuracy
- Use `few_shot` for complex iconographic content
- Review and iterate on empty or unexpected classifications
- Consider manual review of low-confidence results

## Additional Resources

- [Iconclass Classification System](https://iconclass.org/)
- [Ollama Documentation](https://ollama.ai/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

## Support

For questions or issues:

- Check the [main README](../README.md)
- Review the [USAGE guide](../USAGE.md)
- Open an issue on [GitHub](https://github.com/Stadt-Geschichte-Basel/iconclass-classification/issues)
