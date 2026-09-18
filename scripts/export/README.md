# Model Export System

This directory contains the complete model export pipeline for the Samsone project. The system exports trained audio-language models for inference using ExecuTorch.

## Overview

The export process converts trained models into optimized formats that can be deployed on edge devices:

1. **AudioModel** - Contains audio preprocessors, audio-language model, embeddings, and argmax components
2. **TextModel** - A standalone language model for processing multimodal embeddings

All export operations are performed through a single script (`export.py`) that handles the entire pipeline.

## Quick Start

To run the complete export pipeline:

```bash
uv run python scripts/export/export.py
```

The script accepts a configuration file. Use the `--config-path` argument or modify the default in the script:

```bash
uv run python scripts/export/export.py --config-name config99M
```

## Configuration

All export parameters are configured through YAML files in the `scripts/export/` directory. Available configurations include:

- `config99M.yaml` - 99M parameter model
- `config134M.yaml` - 134M parameter model
- `config356M.yaml` - 356M parameter model

The configuration includes:
- Audio export settings (input shapes, output paths)
- Audio processor configuration (feature size, sampling rate, etc.)
- Text model export configuration (automatically generates `export_llm_config.yaml`)

## Export Pipeline

The export process performs the following steps automatically:

### 1. Export AudioModel
- Loads the trained audio-language model
- Exports multiple components as a single ExecuTorch file:
  - `forward`: Audio-language model
  - `embedding`: Text model embeddings
  - `argmax`: Token selection
  - `processor`: Audio preprocessor (mel spectrogram)

### 2. Convert Text Model Weights
- Saves the text model separately
- Converts weights to ExecuTorch-compatible format

### 3. Export TextModel
- Uses the generated `export_llm_config.yaml`
- Exports the text model as a standalone ExecuTorch file

## Output Files

After successful export, the following files are generated:

- **AudioModel** (`outputs/export/exported/audio_lm.pte` or path specified in config) - Combined audio processing model with all components
- **TextModel** (location specified in `export_llm_config.yaml`) - Standalone language model for text generation

## Architecture Details

### AudioModel
The AudioModel combines four components into a single ExecuTorch file:

1. **forward** - The main audio-language model that processes audio features and text tokens
2. **embedding** - Extracts embeddings from text tokens using the text model's embedding layer
3. **argmax** - Selects the most likely token from model logits
4. **processor** - Converts raw audio waveforms to mel spectrograms

### TextModel
The TextModel is exported separately and handles text generation. It receives multimodal embeddings from the AudioModel and generates text responses.

## Prerequisites

Before running the export script, you must modify the ExecuTorch model factory:

Add the following code to `.venv/lib/python3.12/site-packages/executorch/examples/models/model_factory.py` in the `create_model()` function:

```python
example_kwarg_inputs = {
    'attn_options': {
        'input_pos': torch.tensor([0, 1, 2], dtype=torch.long),
    },
    'h': torch.randn((1,3,576)),  # tokens, with kv cache our input token length is always just 1 token.,
}
dynamic_shapes = (
        {'input_pos': {0: torch.export.Dim('token_dim', max=500)}},
        {1: torch.export.Dim('token_dim', max=500)},
        
)
example_inputs = model.get_example_inputs()
example_inputs = tuple()
return (
    model.get_eager_model(),
    example_inputs,
    example_kwarg_inputs,
    dynamic_shapes,
)
```

in `.venv/lib/python3.12/site-packages/executorch/examples/models/llama/attention.py` at line 486:

change:
```python
if self.enable_dynamic_shape:
```

to:
```python
if self.enable_dynamic_shape and len(input_pos)==1:
```

in `.venv/lib/python3.12/site-packages/executorch/examples/models/llama/llama_transformer.py`:

change:
```python
freqs_cos, freqs_sin = self.rope.get_freqs(attn_options.get('input_pos'), seqlen)
```

to:
```python
freqs_cos, freqs_sin = self.rope.get_freqs_using_indices(attn_options.get('input_pos'))
```

## Customization

### Changing Model Configuration
To export a different model, use the appropriate config file:

```bash
uv run python scripts/export/export.py --config-name config356M
```

Or modify the `audio_export.config_path` in your config file to point to your model configuration.

### Modifying Output Paths
All output paths can be customized through the configuration variables:
- `audio_export.output_file` - Path for the AudioModel ExecuTorch file
- `text_model_dir` - Directory for text model components
- `export_llm_config_path` - Path for generated text model export config

### Adjusting Input Shapes
The input tensor shapes can be modified in the configuration:
- `audio_export.audio_features_shape` - Audio input dimensions (default: `[2, 80, 3000]`)
- `audio_export.post_audio_tokens_shape` - Text input dimensions (default: `[1, 50]`)
- `audio_export.max_new_tokens` - Maximum tokens for generation (used in dynamic shapes)

### Audio Processor Configuration
Modify the audio processor settings in the `audio_processor` section:
- `feature_size` - Mel spectrogram feature size (default: 80)
- `sampling_rate` - Audio sampling rate (default: 16000 Hz)
- `hop_length` - STFT hop length (default: 160)
- `chunk_length` - Audio chunk length in seconds (default: 30)
- `n_fft` - FFT size (default: 400)
- `stack_output` - Whether to stack output (default: false)

## Troubleshooting

### Debugging

To run the export with more visibility:

```bash
cd scripts/export
python export.py --config-name config99M
```

The script outputs progress messages for each step:
1. Model initialization
2. Text model saving
3. Model exporting (AudioModel components)
4. Edge compilation and lowering
5. Text model weight conversion
6. Text model export

### Common Issues

**Export failures**: Ensure the ExecuTorch modifications in the Prerequisites section are applied correctly.

**Out of memory errors**: Reduce the `audio_export.audio_features_shape` dimensions or use a smaller model configuration.

**Dynamic shape errors**: Verify that the dynamic shape dimensions in `export_models()` match your expected input sizes.