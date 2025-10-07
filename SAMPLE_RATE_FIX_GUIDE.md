# Sample Rate Fix - Quick Reference

## ⚡ Quick Usage

### Option 1: Automatic (Recommended)
```python
from voice_ai import create_processor

processor = create_processor()

# Just prepare training data - validation is automatic!
training_metadata = processor.prepare_training_data(
    source_dir="audio_samples",
    output_dir="F5_TTS/finetune_data",
    speaker_name="my_speaker"
    # validate_sample_rate=True is default!
)
```

### Option 2: Fix Existing Data
```python
# Fix already prepared training data
result = processor.fix_sample_rate(
    training_dir="F5_TTS/finetune_data",
    target_rate=24000,
    backup=True
)

print(f"Status: {result['status']}")  # 'ok' or 'fixed'
print(f"Files checked: {result['files_checked']}")
print(f"Files fixed: {result.get('files_fixed', 0)}")
```

## 🎯 What Was Fixed

| Component | Issue | Fix |
|-----------|-------|-----|
| **voice_trainer.py** | Hardcoded 16kHz resampling | Changed to 24kHz |
| **prepare_training_data()** | No validation | Auto-validates after processing |
| **New Method** | Missing validation tool | Added `validate_and_fix_sample_rate()` |
| **processor.py** | No public API | Added `fix_sample_rate()` method |

## 🔧 Integration Points

### 1. Voice Trainer (`voice_ai/voice_trainer.py`)
```python
# New method
def validate_and_fix_sample_rate(training_dir, target_rate=24000, backup=True)

# Updated method
def prepare_training_data(..., validate_sample_rate=True)
```

### 2. Processor (`voice_ai/processor.py`)
```python
# New public method
def fix_sample_rate(training_dir, target_rate=24000, backup=True)
```

## 📊 Return Values

```python
{
    "status": "ok" | "fixed",
    "target_rate": 24000,
    "files_checked": 32,
    "files_fixed": 0,
    "backup_dir": "path/to/backup" | None
}
```

## ✅ Checklist for Existing Projects

- [ ] Run `processor.fix_sample_rate()` on training data
- [ ] Verify all files at 24kHz
- [ ] Optionally retrain model with corrected data
- [ ] Test synthesis quality

## 🚀 Checklist for New Projects

- [ ] Use `processor.prepare_training_data()` (auto-validates)
- [ ] Check validation results in metadata
- [ ] Start synthesis - no extra steps needed!

## 🎵 Quality Settings (Bonus)

```python
# Generate high-quality audio
processor.synthesize_speech(
    text="Your text here",
    reference_audio="sample.wav",
    nfe_step=64,        # 32=fast, 64-96=high quality
    cfg_strength=2.5,   # 2.0=default, 3.0-4.0=clearer
    output_path="output.wav"
)
```

## 📁 Files Created/Modified

### New Files
- `test_sample_rate_fix.py` - Test script
- `demo_updated_pipeline.py` - Complete demo
- `SAMPLE_RATE_INTEGRATION.md` - Full documentation
- `SAMPLE_RATE_FIX_GUIDE.md` - This quick reference

### Modified Files
- `voice_ai/voice_trainer.py` - Core fix + validation method
- `voice_ai/processor.py` - Public API
- `voice_ai/README.md` - Documentation

## 💡 Key Insights

1. **Root Cause**: 16kHz training data vs 24kHz model requirement
2. **Symptom**: Gibberish/distorted speech output
3. **Fix**: Resample all training data to 24kHz
4. **Prevention**: Automatic validation in pipeline
5. **Safety**: Backups created before any changes

## 🆘 Troubleshooting

### Still Getting Gibberish?
1. Check sample rate: `processor.fix_sample_rate(...)`
2. Retrain model with corrected data
3. Verify synthesis settings (nfe_step, cfg_strength)
4. Check training data quality (no filler words)

### Need to Disable Auto-Validation?
```python
training_metadata = processor.prepare_training_data(
    source_dir="audio_samples",
    validate_sample_rate=False  # Not recommended!
)
```

## 📞 Support

See full documentation:
- `SAMPLE_RATE_INTEGRATION.md` - Complete integration guide
- `FIXING_GIBBERISH.md` - General quality troubleshooting
- `voice_ai/README.md` - Module documentation
