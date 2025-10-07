# Voice Quality Improvements Guide

## 🎯 Overview
The voice synthesis now automatically uses your **fine-tuned model** instead of the base model, with configurable quality settings.

## ✨ What's New

### 1. **Automatic Fine-tuned Model Detection**
- The synthesizer now automatically finds and uses your fine-tuned checkpoint
- No need to manually specify checkpoint paths
- Falls back to base model if no checkpoint is found

### 2. **Quality Control Parameters**
You can now control audio quality with these parameters:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `nfe_step` | 32 | 16-128 | Denoising steps (higher = better quality, slower) |
| `cfg_strength` | 2.0 | 0.5-4.0 | Guidance strength (higher = follows text more closely) |
| `sway_sampling_coef` | -1.0 | -1.0 to 1.0 | Sampling coefficient (affects variance) |
| `speed` | 1.0 | 0.5-2.0 | Speech speed multiplier |
| `checkpoint_path` | Auto | Path or None | Override auto-detected checkpoint |

### 3. **Quality Presets**

#### Fast (Good for Testing)
```python
processor.synthesize_speech(
    text="Your text here",
    reference_audio="path/to/reference.wav",
    nfe_step=16,
    cfg_strength=1.5
)
```

#### Standard (Balanced)
```python
processor.synthesize_speech(
    text="Your text here", 
    reference_audio="path/to/reference.wav",
    nfe_step=32,
    cfg_strength=2.0
)
```

#### High Quality (Recommended)
```python
processor.synthesize_speech(
    text="Your text here",
    reference_audio="path/to/reference.wav", 
    nfe_step=64,
    cfg_strength=2.5
)
```

#### Maximum Quality (Slow)
```python
processor.synthesize_speech(
    text="Your text here",
    reference_audio="path/to/reference.wav",
    nfe_step=128,
    cfg_strength=3.0
)
```

## 🚀 Usage Examples

### Using Fine-tuned Model (Automatic)
```python
from voice_ai import create_processor

processor = create_processor(workspace_path="voice_clone_project_20251005_203311")

# Automatically uses fine-tuned model with high quality
audio = processor.synthesize_speech(
    text="This will use the fine-tuned model automatically.",
    reference_audio="path/to/reference.wav",
    nfe_step=64,  # High quality
    cfg_strength=2.5
)
```

### Force Base Model
```python
# Explicitly use base model by setting checkpoint to None
audio = processor.synthesize_speech(
    text="This uses the base pre-trained model.",
    reference_audio="path/to/reference.wav",
    checkpoint_path=None,  # Force base model
    nfe_step=32
)
```

### Custom Checkpoint
```python
# Use a specific checkpoint file
audio = processor.synthesize_speech(
    text="Using custom checkpoint.",
    reference_audio="path/to/reference.wav",
    checkpoint_path="/path/to/custom_model.pt",
    nfe_step=64
)
```

## 🧪 Testing Quality

Run the comparison test script:
```bash
python test_voice_quality.py
```

This generates 4 audio files for comparison:
1. Base model (low quality)
2. Base model (high quality)  
3. Fine-tuned model (low quality)
4. Fine-tuned model (high quality)

## 📊 Quality vs Speed Tradeoff

| NFE Steps | Quality | Speed | Use Case |
|-----------|---------|-------|----------|
| 16 | Low | Very Fast | Quick tests, previews |
| 32 | Good | Fast | Standard generation |
| 64 | High | Medium | Production quality |
| 96 | Very High | Slow | Critical quality needs |
| 128 | Maximum | Very Slow | Maximum quality output |

## 🔧 Troubleshooting

### Audio Sounds Like Gibberish
- **Cause:** Using base model instead of fine-tuned model
- **Fix:** Ensure fine-tuned checkpoint exists and is auto-detected
- **Check:** Look for "🔬 Checkpoint: model_last.pt" in output

### Audio Quality is Poor
- **Try:** Increase `nfe_step` to 64 or 128
- **Try:** Adjust `cfg_strength` between 2.0-3.0
- **Try:** Use a different reference audio sample

### Synthesis is Too Slow
- **Reduce:** `nfe_step` to 16 or 32
- **Note:** Quality will decrease but speed increases significantly

## 📁 Output Location
Generated audio is saved in:
```
voice_clone_project_XXXXXXXX_XXXXXX/
└── 07_generated_audio/
    ├── tests/                    # Standard tests
    ├── tests_finetuned/          # High-quality tests  
    ├── comparison/               # Quality comparison
    └── custom_phrase.wav         # Custom synthesis
```

## 💡 Best Practices

1. **Always use the fine-tuned model** for production (it's now the default)
2. **Start with nfe_step=64** for good quality/speed balance
3. **Use high-quality reference audio** (clear speech, low noise)
4. **Test with different cfg_strength** values to find sweet spot
5. **Keep reference audio 10-20 seconds** long for best results

## 🎯 Next Steps

1. Listen to comparison outputs in `07_generated_audio/comparison/`
2. Choose your preferred quality settings
3. Run full pipeline with new settings
4. Fine-tune more if needed with additional training data
