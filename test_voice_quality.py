#!/usr/bin/env python3
"""
Test Voice Quality with Different Settings
Compare base model vs fine-tuned model with various quality parameters
"""

from voice_ai import create_processor
from pathlib import Path
import sys

def main():
    # Configuration
    PROJECT_DIR = "voice_clone_project_20251005_203311"
    REFERENCE_AUDIO = f"{PROJECT_DIR}/04_voice_filtered/left_speaker/segment_010_17.5s.wav"
    TEST_TEXT = "Hello, this is a test of the fine-tuned voice model with high quality synthesis."
    
    # Create processor
    print("🎵 Initializing Voice AI Processor...\n")
    processor = create_processor(workspace_path=PROJECT_DIR)
    
    # Test configurations
    test_configs = [
        {
            "name": "Base Model (Low Quality)",
            "output_dir": f"{PROJECT_DIR}/07_generated_audio/comparison/base_low",
            "checkpoint_path": None,  # Use base model
            "nfe_step": 32,
            "cfg_strength": 2.0
        },
        {
            "name": "Base Model (High Quality)",
            "output_dir": f"{PROJECT_DIR}/07_generated_audio/comparison/base_high",
            "checkpoint_path": None,  # Use base model
            "nfe_step": 64,
            "cfg_strength": 2.5
        },
        {
            "name": "Fine-tuned Model (Low Quality)",
            "output_dir": f"{PROJECT_DIR}/07_generated_audio/comparison/finetuned_low",
            "nfe_step": 32,
            "cfg_strength": 2.0
            # Auto-detects fine-tuned checkpoint
        },
        {
            "name": "Fine-tuned Model (High Quality)",
            "output_dir": f"{PROJECT_DIR}/07_generated_audio/comparison/finetuned_high",
            "nfe_step": 64,
            "cfg_strength": 2.5
            # Auto-detects fine-tuned checkpoint
        }
    ]
    
    print("🧪 Running Voice Quality Comparison Tests\n")
    print(f"Test phrase: '{TEST_TEXT}'\n")
    print("=" * 80)
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n📊 Test {i}/4: {config['name']}")
        print(f"   NFE Steps: {config['nfe_step']}")
        print(f"   CFG Strength: {config['cfg_strength']}")
        
        if config.get('checkpoint_path') is None and 'Base Model' in config['name']:
            print(f"   Model: Base (pre-trained)")
        else:
            print(f"   Model: Fine-tuned (custom)")
        
        output_path = f"{config['output_dir']}/test_output.wav"
        
        # Generate audio
        result = processor.synthesize_speech(
            text=TEST_TEXT,
            reference_audio=REFERENCE_AUDIO,
            output_path=output_path,
            **{k: v for k, v in config.items() if k not in ['name', 'output_dir']}
        )
        
        if result:
            file_size = Path(result).stat().st_size / 1024  # KB
            print(f"   ✅ Generated: {Path(result).name} ({file_size:.1f} KB)")
        else:
            print(f"   ❌ Failed to generate audio")
    
    print("\n" + "=" * 80)
    print("\n📁 Comparison Results Location:")
    print(f"   {PROJECT_DIR}/07_generated_audio/comparison/")
    print("\n💡 Tips:")
    print("   • Base Model: Uses pre-trained F5-TTS (may sound generic)")
    print("   • Fine-tuned Model: Uses your custom trained model (should sound like target voice)")
    print("   • NFE Steps: Higher = better quality but slower (32 = fast, 64 = high quality, 128 = max)")
    print("   • CFG Strength: Controls how closely it follows the text (2.0 = standard, 2.5-3.0 = stronger)")
    print("\n🎧 Listen to the files to compare quality!")

if __name__ == "__main__":
    main()
