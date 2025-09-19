#!/usr/bin/env python3
"""
F5-TTS Voice Cloning Test Script
Test voice cloning with the prepared left_speaker samples
"""

import os
import sys
from pathlib import Path
import torch
import torchaudio
import soundfile as sf

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from f5_tts.api import F5TTS
    F5TTS_AVAILABLE = True
    print("✅ F5-TTS imported successfully")
except ImportError as e:
    print(f"❌ F5-TTS import failed: {e}")
    F5TTS_AVAILABLE = False
    sys.exit(1)

class VoiceCloningTest:
    """Test voice cloning with F5-TTS"""
    
    def __init__(self, samples_dir: str):
        self.samples_dir = Path(samples_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🎯 Voice Cloning Test")
        print(f"📁 Samples: {self.samples_dir}")
        print(f"🖥️  Device: {self.device}")
        
        # Initialize F5-TTS
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load F5-TTS model"""
        try:
            print("🔥 Loading F5-TTS model...")
            self.model = F5TTS(
                model="F5TTS_Base",  # Model name
                ckpt_file="",        # Use default pretrained
                vocab_file="",       # Use default
                device=self.device
            )
            print("✅ F5-TTS model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading F5-TTS: {e}")
            print("🔧 Trying alternative initialization...")
            try:
                # Try simpler initialization
                self.model = F5TTS()
                print("✅ F5-TTS model loaded with defaults")
            except Exception as e2:
                print(f"❌ Alternative loading also failed: {e2}")
                self.model = None
    
    def get_reference_audio(self):
        """Get the best reference audio sample"""
        audio_files = list(self.samples_dir.glob("*.wav"))
        if not audio_files:
            raise FileNotFoundError(f"No audio files found in {self.samples_dir}")
        
        # Use the first sample as reference
        ref_file = audio_files[0]
        print(f"🎤 Using reference: {ref_file.name}")
        
        # Load audio
        audio, sr = torchaudio.load(str(ref_file))
        if sr != 24000:
            resampler = torchaudio.transforms.Resample(sr, 24000)
            audio = resampler(audio)
        
        return audio.squeeze(), str(ref_file)
    
    def test_voice_cloning(self, text: str = "Hello, this is a test of voice cloning using F5-TTS."):
        """Test voice cloning with given text"""
        if not self.model:
            print("❌ Model not loaded")
            return None
        
        try:
            # Get reference audio
            ref_audio, ref_file = self.get_reference_audio()
            
            # Reference text (simplified for testing)
            ref_text = "This is a voice sample for training the speech synthesis model."
            
            print(f"🎬 Generating speech...")
            print(f"   Text: '{text}'")
            print(f"   Reference: {Path(ref_file).name}")
            
            # Generate speech using F5-TTS inference
            try:
                # Method 1: Try the new API
                output_audio, sample_rate = self.model.infer(
                    ref_audio=ref_audio.unsqueeze(0),
                    ref_text=ref_text,
                    gen_text=text,
                    target_sample_rate=24000,
                    cross_fade_duration=0.15
                )
            except AttributeError:
                # Method 2: Try alternative method
                output_audio = self.model.sample(
                    text=text,
                    ref_audio=ref_audio,
                    ref_text=ref_text
                )
                sample_rate = 24000
            
            # Save output
            output_path = Path("generated_voice_test.wav")
            sf.write(str(output_path), output_audio.squeeze().cpu().numpy(), sample_rate)
            
            print(f"✅ Voice cloning successful!")
            print(f"💾 Generated audio saved: {output_path}")
            print(f"🎵 Duration: {len(output_audio.squeeze()) / sample_rate:.1f} seconds")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Voice cloning failed: {e}")
            return None
    
    def test_multiple_samples(self):
        """Test with multiple text samples"""
        test_texts = [
            "Hello, this is a test of the voice cloning system.",
            "The quick brown fox jumps over the lazy dog.",
            "I am testing the quality of the synthesized speech.",
            "This voice should sound like the original speaker."
        ]
        
        print(f"🎬 Testing multiple text samples...")
        
        for i, text in enumerate(test_texts):
            print(f"\n📝 Test {i+1}: '{text[:30]}...'")
            output_path = self.test_voice_cloning(text)
            
            if output_path:
                # Rename with test number
                new_name = f"generated_test_{i+1}.wav"
                output_path.rename(new_name)
                print(f"   💾 Saved as: {new_name}")
            else:
                print(f"   ❌ Failed")

def main():
    # Test with left_speaker samples
    samples_dir = "/Users/adamleeperelman/Documents/LLM/Voice Clone/ai_voice_samples/left_speaker"
    
    print("🎯 F5-TTS Voice Cloning Test")
    
    # Initialize tester
    tester = VoiceCloningTest(samples_dir)
    
    if not tester.model:
        print("❌ Cannot proceed without model")
        return 1
    
    # Run single test
    print("\n🔬 Single test:")
    result = tester.test_voice_cloning()
    
    if result:
        print("\n🔬 Multiple tests:")
        tester.test_multiple_samples()
        
        print(f"\n🎉 Voice cloning tests complete!")
        print(f"📁 Check the generated *.wav files in the current directory")
    else:
        print("❌ Voice cloning test failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
