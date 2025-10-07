#!/usr/bin/env python3
"""
Updated Voice Cloning Pipeline Demo
Shows the integrated sample rate validation in action
"""

from voice_ai import create_processor

def demo_updated_pipeline():
    """Demonstrate the updated pipeline with automatic sample rate validation"""
    
    print("🎯 Voice Cloning Pipeline - Updated with Sample Rate Validation")
    print("=" * 80)
    
    # Create processor
    processor = create_processor(workspace_path='voice_clone_project_20251005_203311')
    
    print("\n📋 What's New:")
    print("   ✅ Automatic 24kHz sample rate validation")
    print("   ✅ Built-in sample rate correction")
    print("   ✅ Prevents gibberish output")
    print("   ✅ Zero configuration required")
    
    # Example 1: Check existing training data
    print("\n" + "=" * 80)
    print("1️⃣  VALIDATE EXISTING TRAINING DATA")
    print("=" * 80)
    
    result = processor.fix_sample_rate(
        training_dir='voice_clone_project_20251005_203311/05_training_data',
        target_rate=24000,
        backup=True
    )
    
    print(f"\n📊 Validation Result: {result.get('status', 'unknown').upper()}")
    print(f"   Files Checked: {result.get('files_checked', 0)}")
    print(f"   Target Rate: {result.get('target_rate', 'N/A')} Hz")
    
    if result.get('status') == 'fixed':
        print(f"   Files Fixed: {result.get('files_fixed', 0)}")
        print(f"   Backup: {result.get('backup_dir', 'N/A')}")
    
    # Example 2: Show how future training prep works
    print("\n" + "=" * 80)
    print("2️⃣  HOW FUTURE TRAINING PREP WORKS")
    print("=" * 80)
    
    print("\n📝 Code Example:")
    print("""
    # Prepare training data (sample rate validation automatic!)
    training_metadata = processor.prepare_training_data(
        source_dir="audio_samples",
        output_dir="F5_TTS/finetune_data",
        speaker_name="my_speaker",
        validate_sample_rate=True  # Default: enabled
    )
    
    # Behind the scenes:
    # 1. Process audio files
    # 2. Save at 24kHz (hardcoded fix)
    # 3. Auto-validate all files
    # 4. Fix any remaining issues
    # 5. Report results in metadata
    """)
    
    # Example 3: Synthesis with quality settings
    print("\n" + "=" * 80)
    print("3️⃣  SYNTHESIS WITH CORRECT SAMPLE RATE")
    print("=" * 80)
    
    print("\n🎤 Generating test audio with optimal settings...")
    
    output_path = 'voice_clone_project_20251005_203311/07_generated_audio/demo_24khz.wav'
    
    result = processor.synthesize_speech(
        text='This is a demonstration of the updated voice cloning pipeline with automatic sample rate validation.',
        reference_audio='voice_clone_project_20251005_203311/05_training_data/wavs/main_speaker_004.wav',
        reference_text='in terms of the cocoa thing look in the end of the day I got for us over 300,000',
        output_path=output_path,
        nfe_step=64,  # High quality
        cfg_strength=2.5  # Balanced clarity
    )
    
    if result:
        import soundfile as sf
        info = sf.info(result)
        print(f"\n✅ Audio Generated Successfully!")
        print(f"   Output: {result}")
        print(f"   Sample Rate: {info.samplerate} Hz ✅")
        print(f"   Duration: {info.duration:.1f}s")
        print(f"   Channels: {info.channels}")
        
        if info.samplerate == 24000:
            print(f"\n🎉 Perfect! Audio at correct 24kHz sample rate")
        else:
            print(f"\n⚠️  Warning: Unexpected sample rate")
    
    # Summary
    print("\n" + "=" * 80)
    print("📚 SUMMARY")
    print("=" * 80)
    
    print("""
✅ Sample Rate Integration Complete!

Key Benefits:
1. Automatic validation during training prep
2. Built-in correction for existing data
3. Prevents gibberish before it happens
4. Zero configuration needed
5. Safety backups for all changes

Usage:
- New projects: Just use prepare_training_data() - it's automatic!
- Existing data: Call processor.fix_sample_rate() once
- Synthesis: Always produces 24kHz output now

Files Modified:
- voice_ai/voice_trainer.py (validation method + 24kHz fix)
- voice_ai/processor.py (public API method)
- voice_ai/README.md (documentation)

Next Steps:
1. Test the generated audio (demo_24khz.wav)
2. If quality is good, you're done!
3. If needed, retrain with corrected 24kHz data
    """)
    
    print("=" * 80)
    print("🎯 Demo Complete!")

if __name__ == "__main__":
    demo_updated_pipeline()
