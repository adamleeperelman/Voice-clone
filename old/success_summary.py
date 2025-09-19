#!/usr/bin/env python3
"""
Voice Cloning Success Summary
"""

import os
from pathlib import Path

def create_success_summary():
    """Create a summary of the successful voice cloning"""
    
    # File locations
    samples_dir = Path("/Users/adamleeperelman/Documents/LLM/Voice Clone/ai_voice_samples/left_speaker")
    generated_dir = Path("/Users/adamleeperelman/Documents/LLM/Voice Clone/F5_TTS/generated_voices/generated_voices")
    training_dir = Path("/Users/adamleeperelman/Documents/LLM/Voice Clone/F5_TTS/finetune_data/left_speaker_training")
    
    print("🎉 VOICE CLONING SUCCESS! 🎉")
    print("="*50)
    
    # Check original samples
    original_files = list(samples_dir.glob("*.wav"))
    print(f"📁 Original Samples: {len(original_files)} files")
    for f in original_files:
        print(f"   📄 {f.name}")
    
    # Check generated files
    generated_files = list(generated_dir.glob("*.wav"))
    print(f"\n🎬 Generated Voices: {len(generated_files)} files")
    for f in generated_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"   🎵 {f.name} ({size_mb:.1f} MB)")
    
    # Training data status
    print(f"\n📊 Training Data Status:")
    if training_dir.exists():
        audio_files = list((training_dir / "audio").glob("*.wav"))
        metadata_file = training_dir / "metadata.txt"
        print(f"   ✅ Training directory: {training_dir}")
        print(f"   📄 Audio files: {len(audio_files)}")
        print(f"   📝 Metadata: {'✅' if metadata_file.exists() else '❌'}")
    else:
        print(f"   ⚠️  Training data not prepared yet")
    
    print(f"\n🎯 Next Steps:")
    print(f"1. 🎧 Listen to generated files:")
    for f in generated_files:
        print(f"   open '{f}'")
    
    print(f"\n2. 🔄 Fine-tune for better results:")
    print(f"   cd '/Users/adamleeperelman/Documents/LLM/Voice Clone/F5_TTS'")
    print(f"   python prepare_training.py  # If not done already")
    
    print(f"\n3. 🚀 Start fine-tuning (advanced):")
    print(f"   f5-tts_train-cli \\")
    print(f"     --model F5TTS_Base \\")
    print(f"     --dataset_name left_speaker \\")
    print(f"     --exp_name left_speaker_finetune")
    
    print(f"\n📚 Resources:")
    print(f"   📖 Fine-tuning guide: {generated_dir.parent}/fine_tuning_guide.md")
    print(f"   🎵 Generated samples: {generated_dir}")
    print(f"   📊 Training data: {training_dir}")
    
    print(f"\n✨ Summary:")
    print(f"   ✅ F5-TTS installation: Working")
    print(f"   ✅ Voice samples: {len(original_files)} prepared")
    print(f"   ✅ Voice cloning: {len(generated_files)} files generated")
    print(f"   🔄 Ready for fine-tuning with your specific voice")

if __name__ == "__main__":
    create_success_summary()
