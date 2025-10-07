# Voice AI Modular System

A comprehensive, modular voice processing system that combines voice separation, training data preparation, and speech synthesis into a unified interface.

## Overview

This system provides a modular architecture for:
- **Voice Separation**: Separate audio recordings into individual speakers using Whisper + GPT analysis
- **Training Data Preparation**: Prepare high-quality training data for F5-TTS voice cloning
- **Voice Synthesis**: Generate speech using F5-TTS with custom or pre-trained models

## Quick Start

```python
from voice_ai import create_processor

# Initialize the processor
processor = create_processor()

# Run the complete pipeline
results = processor.full_pipeline(
    input_audio="path/to/recording.mp3",
    test_text="Hello, this is a test of voice cloning.",
    speaker_name="my_speaker"
)
```

## Architecture

### Main Components

1. **VoiceAIProcessor** (`voice_ai/processor.py`)
   - Main orchestrator class for all operations
   - Unified interface for the complete pipeline
   - Provides high-level methods for common tasks

2. **VoiceAISeparator** (`voice_ai/voice_separator.py`)
   - Separates audio by speakers using Whisper transcription
   - Quality filtering with adaptive thresholds
   - Optional GPT-4 enhanced speaker analysis

3. **VoiceTrainer** (`voice_ai/voice_trainer.py`)
   - Prepares training data for F5-TTS
   - Validates data quality and compatibility
   - Manages fine-tuning process

4. **VoiceSynthesizer** (`voice_ai/voice_synthesizer.py`)
   - Generates speech using F5-TTS
   - Supports batch processing and testing
   - Handles multiple model types

5. **Module Init** (`voice_ai/__init__.py`)
   - Simple import orchestrator
   - Makes all classes available at module level

## Usage Examples

### Voice Separation

```python
from voice_ai import VoiceAIProcessor

processor = VoiceAIProcessor()

# Separate audio into speakers
separated_files = processor.separate_voices(
    input_path="recording.mp3",
    output_dir="separated_audio",
    min_duration=8.0,
    min_confidence=-0.5
)
```

### Training Data Preparation

```python
# Prepare training data from separated audio
training_metadata = processor.prepare_training_data(
    source_dir="separated_audio/left_speaker",
    output_dir="F5_TTS/finetune_data",
    speaker_name="custom_speaker",
    validate_sample_rate=True  # Auto-validates and fixes sample rates
)

# Validate the training data
validation = processor.validate_training_data("F5_TTS/finetune_data")
print(f"Training samples: {validation['total_samples']}")
print(f"Issues found: {len(validation['issues'])}")

# Fix sample rate issues (if needed separately)
sample_rate_fix = processor.fix_sample_rate(
    training_dir="F5_TTS/finetune_data",
    target_rate=24000,  # F5-TTS requires 24kHz
    backup=True  # Backup original files
)
```

### Voice Synthesis

```python
# Generate speech from text
audio_file = processor.synthesize_speech(
    text="Hello, this is synthesized speech.",
    reference_audio="separated_audio/left_speaker/sample.wav",
    model="F5-TTS"
)

# Run voice cloning tests
test_files = processor.test_voice_cloning(
    reference_audio="separated_audio/left_speaker/sample.wav"
)

# Batch synthesis
texts = [
    "First sentence to synthesize.",
    "Second sentence for voice cloning.",
    "Third sentence with the same voice."
]
generated_files = processor.batch_synthesize(texts)
```

### Direct Component Usage

```python
from voice_ai import VoiceAISeparator, VoiceTrainer, VoiceSynthesizer

# Use components individually
separator = VoiceAISeparator(openai_api_key="your-key")
trainer = VoiceTrainer(workspace_path="/path/to/workspace")
synthesizer = VoiceSynthesizer(workspace_path="/path/to/workspace")

# Separate voices
files = separator.separate_audio("input.mp3", "output_dir")

# Prepare training data
metadata = trainer.prepare_training_data("samples_dir", "training_dir")

# Generate speech
audio = synthesizer.generate_audio("Hello world", "reference.wav")
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: For enhanced speaker separation using GPT-4
- Set in your environment or pass directly to the processor

### Parameters

#### Voice Separation
- `min_duration`: Minimum segment length (default: 8.0s)
- `min_confidence`: Minimum Whisper confidence (default: -0.5)
- `silence_thresh`: Silence detection threshold (default: -35dB)
- `min_segment_len`: Minimum segment length for optimization (default: 8.0s)
- `max_segment_len`: Maximum segment length for optimization (default: 15.0s)

#### Training Data Preparation
- `speaker_name`: Name for the custom speaker (default: "custom_speaker")
- `validate_sample_rate`: Auto-validate and fix sample rates (default: True)
- Audio files should be 3-30 seconds long for optimal results
- **Important**: F5-TTS requires 24kHz sample rate - automatic validation prevents gibberish output

#### Voice Synthesis
- `model`: F5-TTS model to use (default: "F5-TTS")
- `speed`: Speech speed multiplier (default: 1.0)
- `remove_silence`: Remove silence from generated audio (default: True)
- `nfe_step`: Denoising steps for quality (32=fast, 64-96=high quality)
- `cfg_strength`: Guidance strength (2.0=default, 3.0-4.0=clearer speech)

## Sample Rate Validation

**Critical Feature**: F5-TTS requires 24kHz sample rate. Mismatched sample rates cause gibberish output.

### Automatic Validation (Recommended)
```python
# Automatically validates during training data preparation
training_metadata = processor.prepare_training_data(
    source_dir="audio_samples",
    validate_sample_rate=True  # Default: enabled
)
```

### Manual Validation
```python
# Fix sample rates in existing training data
result = processor.fix_sample_rate(
    training_dir="F5_TTS/finetune_data",
    target_rate=24000,  # F5-TTS requirement
    backup=True  # Backs up originals before conversion
)

# Result includes:
# - status: 'ok' (no changes) or 'fixed' (files resampled)
# - files_checked: Total files validated
# - files_fixed: Number of files resampled
# - backup_dir: Location of backup files
```

**What it does:**
1. ✅ Checks all audio files in training directory
2. ✅ Identifies files with incorrect sample rates
3. ✅ Creates backup of original files (optional)
4. ✅ Resamples audio to 24kHz using high-quality librosa
5. ✅ Validates conversion success


## Dependencies

- **whisper**: Audio transcription
- **f5-tts**: Voice synthesis
- **pydub**: Audio processing
- **openai**: Enhanced speaker analysis (optional)
- **pathlib**: File system operations

## Installation

```bash
# Install dependencies
pip install openai-whisper f5-tts pydub openai

# For audio format support
pip install ffmpeg-python

# Set OpenAI API key (optional)
export OPENAI_API_KEY="your-api-key"
```

## File Structure

```
voice_ai/
├── __init__.py          # Simple import orchestrator  
├── processor.py         # Main VoiceAIProcessor class
├── voice_separator.py   # VoiceAISeparator component
├── voice_trainer.py     # VoiceTrainer component
├── voice_synthesizer.py # VoiceSynthesizer component
└── README.md           # Documentation

# Generated during processing
ai_voice_samples/
├── left_speaker/        # Separated audio files
├── right_speaker/       # Separated audio files
└── analysis/           # Speaker analysis data

F5_TTS/
└── finetune_data/      # Training data for F5-TTS
    ├── wavs/           # Audio files
    ├── metadata.json   # Training metadata
    └── filelist.txt    # F5-TTS format file list

generated_audio/        # Synthesized audio output
voice_tests/           # Voice cloning test results
```

## Error Handling

The system includes comprehensive error handling:
- Adaptive quality thresholds if initial filtering is too strict
- Graceful fallbacks when optional features (GPT analysis) fail
- Detailed logging and status reporting
- Validation of training data before fine-tuning

## Examples

See `example_usage.py` for a complete demonstration of the modular system.

## Status Checking

```python
# Get comprehensive status
status = processor.get_workspace_status()

print(f"Workspace: {status['workspace_path']}")
print(f"Separated files: {status['separation']['left_speaker_files']}")
print(f"Training samples: {status['training']['total_samples']}")
print(f"F5-TTS available: {status['synthesis']['f5_tts_available']}")
```

## Migration from Original Scripts

The modular system replaces these original scripts:
- `voice_ai_separator.py` → `voice_ai.VoiceAISeparator`
- `F5_TTS/prepare_training.py` → `voice_ai.VoiceTrainer`
- `F5_TTS/working_voice_test.py` → `voice_ai.VoiceSynthesizer`

All functionality is preserved while providing a cleaner, more maintainable interface.
