#!/usr/bin/env python3
"""
Direct Voice Separation Test
Test voice separation directly with pipeline inputs
"""

import os
from pathlib import Path
from voice_ai.voice_separator import VoiceSeparator

def main():
    print("🔧 Direct Voice Separation Test")
    print("=" * 40)
    
    # Use the exact same paths as the pipeline
    project_dir = "/Users/adamleeperelman/Documents/LLM/Voice Clone/voice_clone_project_20250925_173925"
    input_audio = f"{project_dir}/02_stereo_channels/recording_0m_to_15m_left_channel.wav"
    output_dir = f"{project_dir}/03_separated_voices"
    
    # Check if input exists
    if not os.path.exists(input_audio):
        print(f"❌ Input audio not found: {input_audio}")
        return
    
    print(f"📁 Input: {input_audio}")
    print(f"📁 Output: {output_dir}")
    
    # Initialize separator
    print("\n🎭 Initializing VoiceSeparator...")
    separator = VoiceSeparator()
    
    # Run separation with working parameters from our test
    print("\n🚀 Running voice separation...")
    try:
        saved_files = separator.separate_audio(
            input_path=input_audio,
            output_dir=output_dir,
            min_duration=1.0,      # Very low minimum duration
            min_confidence=-2.0,   # Very permissive confidence threshold
            silence_len=1500,      # Shorter silence detection
            min_segment_len=1.0,   # Minimum 1 second
            max_segment_len=20.0,  # Maximum 20 seconds
            max_no_speech_prob=2.0  # Accept ALL segments
        )
        
        print(f"\n✅ SUCCESS! Saved {len(saved_files)} files:")
        for file in saved_files:
            print(f"   📄 {os.path.basename(file)}")
            
        # List what's actually in the directory
        print(f"\n📂 Contents of {output_dir}:")
        if os.path.exists(output_dir):
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isdir(item_path):
                    file_count = len([f for f in os.listdir(item_path) if f.endswith('.wav')])
                    print(f"   📁 {item}/ ({file_count} files)")
                else:
                    print(f"   📄 {item}")
        
    except Exception as e:
        print(f"❌ Error during separation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
