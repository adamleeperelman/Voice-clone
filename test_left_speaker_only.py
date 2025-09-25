#!/usr/bin/env python3
"""
Test script for left speaker only processing
"""

from voice_ai import create_processor
from pathlib import Path
import os

def test_left_speaker_extraction():
    """Test extracting voice segments from left channel only"""
    
    # Use the existing project data
    project_dir = Path("voice_clone_project_20250925_174749")
    if not project_dir.exists():
        print("❌ Project directory not found")
        return
    
    # Change to project directory
    os.chdir(str(project_dir))
    
    # Initialize processor
    processor = create_processor()
    
    # Get the left channel file
    left_channel_file = "02_stereo_channels/recording_0m_to_15m_left_channel.wav"
    
    if not os.path.exists(left_channel_file):
        print(f"❌ Left channel file not found: {left_channel_file}")
        return
    
    print(f"🎯 Processing left channel: {left_channel_file}")
    
    # Create output directory
    output_dir = "03_left_speaker_segments"
    
    try:
        # Extract voice segments
        segments = processor.extract_voice_segments(
            input_path=left_channel_file,
            output_dir=output_dir,
            min_duration=2.0,
            min_confidence=-1.0,
            max_no_speech_prob=1.0,
            silence_len=1000,
            min_segment_len=2.0,
            max_segment_len=20.0
        )
        
        print(f"\n✅ Success! Generated {len(segments)} voice segments")
        for i, segment_file in enumerate(segments[:5], 1):  # Show first 5
            print(f"   {i}. {Path(segment_file).name}")
        
        if len(segments) > 5:
            print(f"   ... and {len(segments) - 5} more")
            
        return segments
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    segments = test_left_speaker_extraction()
