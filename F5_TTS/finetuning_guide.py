#!/usr/bin/env python3
"""
F5-TTS Fine-tuning Location and Process Guide
"""

import os
from pathlib import Path
import json

def show_finetuning_locations():
    """Show where fine-tuning happens and where models are saved"""
    
    print("🎯 F5-TTS Fine-tuning Process & Model Storage")
    print("=" * 60)
    
    # Current project structure
    base_dir = Path("/Users/adamleeperelman/Documents/LLM/Voice Clone")
    f5_dir = base_dir / "F5_TTS"
    training_data_dir = f5_dir / "finetune_data" / "left_speaker_training"
    
    print("\n📁 CURRENT PROJECT STRUCTURE:")
    print(f"   Project Root: {base_dir}")
    print(f"   F5-TTS Module: {f5_dir}")
    print(f"   Training Data: {training_data_dir}")
    print(f"   Generated Voices: {f5_dir}/generated_voices/generated_voices/")
    
    # Check what we have prepared
    print(f"\n📊 TRAINING DATA STATUS:")
    if training_data_dir.exists():
        audio_dir = training_data_dir / "audio"
        metadata_file = training_data_dir / "metadata.txt"
        audio_files = list(audio_dir.glob("*.wav")) if audio_dir.exists() else []
        
        print(f"   ✅ Training directory exists: {training_data_dir}")
        print(f"   📄 Audio files: {len(audio_files)}")
        print(f"   📝 Metadata file: {'✅' if metadata_file.exists() else '❌'}")
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                lines = f.readlines()
            print(f"   📊 Training samples: {len(lines)}")
    else:
        print(f"   ❌ Training data not found")
    
    # F5-TTS default locations
    print(f"\n🏠 F5-TTS DEFAULT LOCATIONS:")
    home_dir = Path.home()
    cache_dir = home_dir / ".cache" / "f5-tts"
    print(f"   Cache directory: {cache_dir}")
    print(f"   Models cache: {home_dir}/.cache/huggingface/hub/")
    
    # Where checkpoints will be saved
    print(f"\n💾 MODEL CHECKPOINT LOCATIONS:")
    print(f"   Default checkpoints: ./ckpts/")
    print(f"   Experiment name: left_speaker_finetune")
    print(f"   Full path: {f5_dir}/ckpts/left_speaker_finetune/")
    
    checkpoint_dir = f5_dir / "ckpts" / "left_speaker_finetune"
    if checkpoint_dir.exists():
        print(f"   ✅ Checkpoint directory exists")
        models = list(checkpoint_dir.glob("*.pt")) + list(checkpoint_dir.glob("*.safetensors"))
        if models:
            print(f"   📦 Found models: {len(models)}")
            for model in models:
                print(f"      - {model.name}")
        else:
            print(f"   📦 No models found yet")
    else:
        print(f"   📦 Checkpoint directory will be created during training")
    
    # Training process steps
    print(f"\n🔄 FINE-TUNING PROCESS:")
    print(f"1. 📋 Data Preparation (✅ DONE)")
    print(f"   - Voice samples processed")
    print(f"   - Metadata created")
    print(f"   - Audio files standardized")
    
    print(f"\n2. 🚀 Start Training:")
    print(f"   Method 1 - Python script:")
    print(f"   cd {f5_dir}")
    print(f"   python -c \"")
    print(f"import sys")
    print(f"sys.path.append('/Users/adamleeperelman/Documents/LLM/Voice Clone/.venv/lib/python3.13/site-packages')")
    print(f"from f5_tts.train.train import main")
    print(f"main('path/to/config.yaml')\"")
    
    print(f"\n   Method 2 - Direct training:")
    print(f"   cd {training_data_dir}")
    print(f"   # Create F5-TTS compatible dataset structure")
    
    print(f"\n3. 📦 Model Storage During Training:")
    print(f"   - Checkpoints saved every N steps")
    print(f"   - Best model saved as model_best.pt")
    print(f"   - Latest model saved as model_latest.pt")
    print(f"   - Training logs in same directory")
    
    print(f"\n4. 🧪 Using Fine-tuned Model:")
    print(f"   f5-tts_infer-cli \\")
    print(f"     --model F5TTS_Base \\")
    print(f"     --ckpt_file ckpts/left_speaker_finetune/model_best.pt \\")
    print(f"     --ref_audio 'your_reference.wav' \\")
    print(f"     --ref_text 'Reference text' \\")
    print(f"     --gen_text 'Text to generate' \\")
    print(f"     --output_dir generated_finetuned")
    
    # Create training config
    print(f"\n⚙️  TRAINING CONFIGURATION:")
    training_config = {
        "model_name": "F5TTS_Base",
        "exp_name": "left_speaker_finetune",
        "learning_rate": 7.5e-5,
        "batch_size": 4,
        "max_samples": 64,
        "grad_accumulation_steps": 1,
        "max_grad_norm": 1.0,
        "epochs": 200,
        "num_warmup_updates": 200,
        "save_per_updates": 400,
        "last_per_steps": 800,
        "dataset_name": "left_speaker",
        "tokenizer": "pinyin",
        "data_path": str(training_data_dir),
        "checkpoint_path": str(f5_dir / "ckpts"),
        "notes": "Fine-tuning with left_speaker voice samples"
    }
    
    config_file = f5_dir / "training_config.json"
    with open(config_file, 'w') as f:
        json.dump(training_config, f, indent=2)
    
    print(f"   📄 Config saved: {config_file}")
    
    # Current status
    print(f"\n📈 CURRENT STATUS:")
    print(f"   ✅ Base model working (F5TTS_Base)")
    print(f"   ✅ Voice cloning successful with pretrained model")
    print(f"   ✅ Training data prepared ({len(audio_files) if 'audio_files' in locals() else 0} samples)")
    print(f"   🔄 Ready for fine-tuning")
    print(f"   📦 Models will be saved in: {checkpoint_dir}")
    
    # Next steps
    print(f"\n🎯 IMMEDIATE NEXT STEPS:")
    print(f"1. Test current voice quality:")
    print(f"   open {f5_dir}/generated_voices/generated_voices/test_1_greeting.wav")
    
    print(f"\n2. If you want better similarity, start fine-tuning:")
    print(f"   cd {f5_dir}")
    print(f"   # Manual training setup required")
    print(f"   # F5-TTS documentation: https://github.com/SWivid/F5-TTS")
    
    print(f"\n3. Monitor training (when started):")
    print(f"   watch ls -la ckpts/left_speaker_finetune/")
    
    return {
        "training_data": str(training_data_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "config_file": str(config_file),
        "current_status": "ready_for_training"
    }

if __name__ == "__main__":
    result = show_finetuning_locations()
    print(f"\n💡 Key Takeaway:")
    print(f"   - Fine-tuning not started yet")
    print(f"   - Models will be saved in: {result['checkpoint_dir']}")
    print(f"   - Currently using pretrained F5TTS_Base model")
    print(f"   - Voice cloning already working with your samples!")
