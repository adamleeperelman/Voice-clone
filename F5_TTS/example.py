#!/usr/bin/env python3
"""
F5-TTS Voice Cloning Example
Demonstrates how to use the voice cloner with generated samples
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from F5_TTS.voice_cloner import VoiceCloner


def main():
    """Main example function"""
    print("🎤 F5-TTS Voice Cloning Example")
    print("="*50)
    
    # Initialize the voice cloner
    # It will automatically look for voice samples in ai_voice_samples/
    try:
        cloner = VoiceCloner(voice_samples_dir="../ai_voice_samples")
    except Exception as e:
        print(f"❌ Error initializing cloner: {e}")
        print("💡 Make sure you've run voice_ai_separator.py first to generate samples")
        return
    
    # List available voices
    voices = cloner.list_voices()
    if not voices:
        print("❌ No voices found!")
        print("💡 Run voice_ai_separator.py first to generate voice samples")
        return
    
    print(f"📋 Available voices: {voices}")
    
    # Use the first available voice
    voice_name = voices[0]
    print(f"🎯 Using voice: {voice_name}")
    
    try:
        # Set the voice (this loads the reference audio)
        cloner.set_voice(voice_name)
        
        # Example texts to synthesize
        example_texts = [
            "Hello, this is a test of the F5-TTS voice cloning system.",
            "The quick brown fox jumps over the lazy dog.",
            "Voice cloning technology is amazing and constantly improving.",
            "Welcome to the future of artificial intelligence and speech synthesis."
        ]
        
        # Create output directory
        output_dir = Path("generated_speech")
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n🗣️  Generating speech samples...")
        
        # Generate speech for each example
        for i, text in enumerate(example_texts, 1):
            print(f"\n{i}. Generating: '{text[:40]}...'")
            
            output_file = output_dir / f"example_{i:02d}_{voice_name}.wav"
            
            try:
                result = cloner.generate_speech(
                    text=text,
                    output_path=str(output_file),
                    speed=1.0
                )
                
                print(f"   ✅ Saved to: {output_file}")
                
            except Exception as e:
                print(f"   ❌ Error generating speech: {e}")
        
        print(f"\n🎉 Example complete!")
        print(f"📁 Check the '{output_dir}' folder for generated audio files")
        
        # Optional: Start interactive mode
        choice = input("\n🤔 Start interactive mode? (y/n): ").strip().lower()
        if choice == 'y':
            cloner.clone_voice_interactive()
    
    except Exception as e:
        print(f"❌ Error during voice cloning: {e}")
        import traceback
        traceback.print_exc()


def quick_test():
    """Quick test function"""
    print("🧪 Quick F5-TTS Test")
    print("="*30)
    
    try:
        cloner = VoiceCloner(voice_samples_dir="../ai_voice_samples")
        voices = cloner.list_voices()
        
        if voices:
            voice_name = voices[0]
            cloner.set_voice(voice_name)
            
            test_text = "This is a quick test."
            audio = cloner.generate_speech(test_text, output_path="quick_test.wav")
            
            print("✅ Quick test successful!")
            print("📁 Generated: quick_test.wav")
        else:
            print("❌ No voices available for testing")
            
    except Exception as e:
        print(f"❌ Quick test failed: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_test()
    else:
        main()
