#!/usr/bin/env python3
"""
Complete Voice Clone Pipeline
Demonstrates the full workflow from audio separation to voice cloning
"""

import os
import sys
import subprocess
from pathlib import Path
import json


def run_command(command, description, cwd=None):
    """Run a command and return success status"""
    print(f"🔧 {description}")
    print(f"   Command: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True,
            cwd=cwd
        )
        print(f"   ✅ Success")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed: {e}")
        if e.stdout:
            print(f"   Output: {e.stdout[:200]}...")
        if e.stderr:
            print(f"   Error: {e.stderr[:200]}...")
        return False, e.stderr


def check_audio_file():
    """Check if the sample audio file exists"""
    audio_file = "Raw Sample/recording.mp3"
    
    if Path(audio_file).exists():
        print(f"✅ Found audio file: {audio_file}")
        return audio_file
    else:
        print(f"❌ Audio file not found: {audio_file}")
        print("💡 Please add an audio file to 'Raw Sample/' directory")
        return None


def run_voice_separation(audio_file):
    """Run the voice separation process"""
    print(f"\n📂 Step 1: Voice Separation")
    print("="*50)
    
    # Check if samples already exist
    samples_dir = Path("ai_voice_samples")
    if samples_dir.exists():
        left_samples = list((samples_dir / "left_speaker").glob("*.wav")) if (samples_dir / "left_speaker").exists() else []
        right_samples = list((samples_dir / "right_speaker").glob("*.wav")) if (samples_dir / "right_speaker").exists() else []
        
        if left_samples or right_samples:
            print(f"🎵 Found existing samples: {len(left_samples)} left, {len(right_samples)} right")
            choice = input("Use existing samples? (y/n): ").strip().lower()
            if choice == 'y':
                return True
    
    # Run voice separation
    success, output = run_command(
        f'python voice_ai_separator.py "{audio_file}"',
        "Running voice separation"
    )
    
    if success:
        # Check results
        left_samples = list((samples_dir / "left_speaker").glob("*.wav")) if (samples_dir / "left_speaker").exists() else []
        right_samples = list((samples_dir / "right_speaker").glob("*.wav")) if (samples_dir / "right_speaker").exists() else []
        
        print(f"🎵 Generated samples: {len(left_samples)} left, {len(right_samples)} right")
        return True
    else:
        print("❌ Voice separation failed")
        return False


def setup_f5tts():
    """Set up F5-TTS dependencies"""
    print(f"\n🤖 Step 2: F5-TTS Setup")
    print("="*50)
    
    f5_dir = Path("F5_TTS")
    if not f5_dir.exists():
        print("❌ F5_TTS directory not found")
        return False
    
    # Check if already set up
    try:
        success, output = run_command(
            "python simple_test.py",
            "Checking F5-TTS setup",
            cwd=f5_dir
        )
        
        if "Dependencies: ✅" in output:
            print("✅ F5-TTS already set up")
            return True
    except:
        pass
    
    # Install dependencies
    choice = input("Install F5-TTS dependencies? (y/n): ").strip().lower()
    if choice != 'y':
        print("⚠️  Skipping F5-TTS setup")
        return False
    
    success, output = run_command(
        "python setup.py",
        "Installing F5-TTS dependencies",
        cwd=f5_dir
    )
    
    return success


def run_voice_cloning():
    """Run voice cloning examples"""
    print(f"\n🗣️  Step 3: Voice Cloning")
    print("="*50)
    
    f5_dir = Path("F5_TTS")
    
    # Run quick test
    success, output = run_command(
        "python example.py --quick",
        "Running voice cloning test",
        cwd=f5_dir
    )
    
    if success:
        print("✅ Voice cloning test successful")
        
        # Offer interactive mode
        choice = input("Run interactive voice cloning? (y/n): ").strip().lower()
        if choice == 'y':
            print("🎤 Starting interactive mode...")
            print("   (This will open an interactive session)")
            
            try:
                subprocess.run(
                    "python example.py",
                    shell=True,
                    cwd=f5_dir
                )
            except KeyboardInterrupt:
                print("\n👋 Interactive session ended")
        
        return True
    else:
        print("❌ Voice cloning failed")
        return False


def show_results():
    """Show the results of the pipeline"""
    print(f"\n📊 Results Summary")
    print("="*50)
    
    # Voice samples
    samples_dir = Path("ai_voice_samples")
    if samples_dir.exists():
        left_samples = list((samples_dir / "left_speaker").glob("*.wav"))
        right_samples = list((samples_dir / "right_speaker").glob("*.wav"))
        
        print(f"🎵 Voice Samples:")
        print(f"   Left speaker: {len(left_samples)} samples")
        print(f"   Right speaker: {len(right_samples)} samples")
        
        # Show sample files
        for i, sample in enumerate(left_samples[:3]):
            print(f"     📁 {sample.name}")
    
    # Generated speech
    generated_dir = Path("F5_TTS/generated_speech")
    if generated_dir.exists():
        generated_files = list(generated_dir.glob("*.wav"))
        print(f"🗣️  Generated Speech: {len(generated_files)} files")
        
        for i, gen_file in enumerate(generated_files[:3]):
            print(f"     🎤 {gen_file.name}")
    
    # Analysis files
    analysis_dir = samples_dir / "analysis"
    if analysis_dir.exists():
        analysis_files = list(analysis_dir.glob("*.json"))
        print(f"📊 Analysis Files: {len(analysis_files)} files")


def main():
    """Main pipeline function"""
    print("🎤 Complete Voice Clone Pipeline")
    print("="*60)
    print("This script will:")
    print("1. Separate voices from stereo audio")
    print("2. Set up F5-TTS for voice cloning")
    print("3. Generate cloned speech samples")
    print("="*60)
    
    # Check prerequisites
    audio_file = check_audio_file()
    if not audio_file:
        return
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  OPENAI_API_KEY not set")
        print("💡 Set with: export OPENAI_API_KEY='your-key-here'")
        choice = input("Continue without OpenAI? (will use basic analysis) (y/n): ").strip().lower()
        if choice != 'y':
            return
    
    try:
        # Step 1: Voice separation
        if not run_voice_separation(audio_file):
            print("❌ Pipeline failed at voice separation")
            return
        
        # Step 2: F5-TTS setup
        if not setup_f5tts():
            print("⚠️  F5-TTS setup skipped or failed")
            print("💡 You can still use the voice samples for other applications")
        else:
            # Step 3: Voice cloning
            run_voice_cloning()
        
        # Show results
        show_results()
        
        print(f"\n🎉 Pipeline Complete!")
        print("="*60)
        print("📁 Check these directories for results:")
        print("   ai_voice_samples/     - Voice samples")
        print("   F5_TTS/generated_speech/ - Cloned speech")
        
    except KeyboardInterrupt:
        print("\n👋 Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
