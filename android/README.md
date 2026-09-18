# Samsone - Audio-LM Chat Application

Samsone is an Android application that provides an audio-enhanced chat interface with on-device AI inference capabilities. The app combines audio processing, text-based chat, and native small audio language models for an intelligent conversational experience.

## Features

- **Audio Processing**: Upload and play audio files using Media3 ExoPlayer
- **AI Chat Interface**: Text-based chat with message bubbles and real-time responses
- **On-Device AI**: Native C++ inference using PyTorch ExecuTorch and tokenizers-cpp
- **Model Integration**: Supports pre-trained .pte model files for audio and text processing
- **Multiple Model Sizes**: Support for 99M, 134M, and 356M parameter models

## Prerequisites

- **Android Studio** Latest stable version
- **Android SDK** API 31+ (minimum), targeting API 35
- **ADB** (Android Debug Bridge) for model deployment
- **Git** for submodule management

## Initial Setup

### 1. Clone and Initialize Submodules

```bash
git clone <repository-url>
cd android
git submodule update --init --recursive
```

This initializes the `tokenizers-cpp` submodule required for native text processing.

### 2. Model Setup

The application requires pre-trained model files (.pte format) to function. The app supports three model sizes: 99M, 134M, and 356M parameters.

1. **Export Models**: Run the model export script for each desired model size
   ```bash
   # From the project root directory
   uv run python scripts/export/export.py --config-name config99M
   uv run python scripts/export/export.py --config-name config134M
   uv run python scripts/export/export.py --config-name config356M
   ```

2. **Push Models to Device**: Use the Gradle task to deploy all models
   ```bash
   ./gradlew pushModelsToDevice
   ```

   This pushes the following model files to `/data/local/tmp/` on the connected device:
   
   **99M Models:**
   - `Samsone99M_audio_model.pte` - Combined audio model (processor, embeddings, argmax)
   - `Samsone99M_text_model.pte` - Text generation model
   
   **134M Models:**
   - `Samsone134M_audio_model.pte` - Combined audio model
   - `Samsone134M_text_model.pte` - Text generation model
   
   **356M Models:**
   - `Samsone356M_audio_model.pte` - Combined audio model
   - `Samsone356M_text_model.pte` - Text generation model

   All models are pushed simultaneously. The application can be configured to use any of the available model sizes at runtime.

### 3. Run the Application

1. Open the project in **Android Studio**
2. Connect an Android device or start an emulator
3. Click **Run** or press `Shift + F10`

Android Studio will handle:
- Dependency resolution
- Native C++ code compilation via CMake
- APK building and installation

## Project Structure

```
app/src/main/
├── java/com/app/samsone/
│   ├── presentation/
│   │   ├── screen/          # Main UI screens
│   │   ├── components/      # Reusable UI components
│   │   ├── models/          # Data models
│   │   └── viewmodel/       # MVVM ViewModels
├── cpp/
│   ├── native-lib.cpp       # JNI bridge
│   └── tokenizers-cpp/      # Text processing library
└── res/                     # Android resources
```

## Technology Stack

- **UI**: Jetpack Compose with Material3
- **Architecture**: MVVM with Hilt dependency injection
- **Audio**: Media3 ExoPlayer
- **ML**: PyTorch ExecuTorch (Android)
- **Native**: C++17 with CMake
- **Networking**: OkHttp

## Model Architecture

The models are structured with two main components:

### AudioModel
A combined ExecuTorch file containing:
- **processor**: Audio preprocessor (converts waveform to mel spectrograms)
- **forward**: Audio-language model (processes audio features and text)
- **embedding**: Text token embeddings
- **argmax**: Token selection from logits

### TextModel
A standalone ExecuTorch file for text generation that receives multimodal embeddings from the AudioModel.

## Model Size Selection

Choose a model size based on your device capabilities and use case:

- **99M**: Fastest inference, suitable for lower-end devices
- **134M**: Balanced performance and quality
- **356M**: Best quality, requires more memory and compute

The export script outputs models to `../outputs/export/{size}/exported/` where `{size}` is `99M`, `134M`, or `356M`.

## Permissions

The app requires the following permissions:
- `INTERNET` - For potential network operations
- `READ_EXTERNAL_STORAGE` - Access audio files
- `READ_MEDIA_AUDIO` - Access audio media on Android 13+

## Troubleshooting

### Build Issues
- Ensure Android Studio is updated to the latest version
- Verify the Android SDK is properly installed
- Check that the device/emulator meets the minimum API requirements

### Model Issues
- Ensure the device is connected via ADB before running `pushModelsToDevice`
- Verify model files exist in `../outputs/export/{size}/exported/` after running the export script
- Check that the device has sufficient storage space for all model variants (~500MB+ total)
- Ensure all three model sizes are exported if you want all variants available

### Native Build Issues
- Run `git submodule update --init --recursive` if C++ compilation fails
- Ensure NDK is installed in Android Studio SDK Manager
- Verify that CMake is properly configured in Android Studio preferences

### Export Issues
- Refer to `scripts/export/README.md` for detailed export troubleshooting
- Ensure ExecuTorch prerequisites are properly configured
- Check that the configuration files in `scripts/export/` match your model setup

## Development Notes

- The app uses Hilt for dependency injection
- Native code is compiled for ARM64, ARMv7, and x86_64 architectures
- Models are loaded from the device's temporary storage (`/data/local/tmp/`) during runtime
- The chat interface supports both text input and audio file processing
- All three model variants can coexist on the device; the active model can be selected at runtime