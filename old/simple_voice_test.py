#!/usr/bin/env python3
"""
Simple F5-TTS Voice Cloning Test
Test voice cloning using F5-TTS CLI interface
"""

import os
import sys
from pathlib import Path
import subprocess
import json

class SimpleVoiceCloner:
    """Simple voice cloning using F5-TTS command line"""
    
    def __init__(self, samples_dir: str):
        self.samples_dir = Path(samples_dir)
        self.output_dir = Path("voice_cloning_output")
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🎯 Simple Voice Cloning Test")
        print(f"📁 Samples: {self.samples_dir}")
        print(f"💾 Output: {self.output_dir}")
    
    def get_reference_sample(self):
        """Get the best reference sample"""
        audio_files = list(self.samples_dir.glob("*.wav"))
        if not audio_files:
            raise FileNotFoundError(f"No audio files found in {self.samples_dir}")
        
        # Use the longest sample as reference (likely highest quality)
        best_file = max(audio_files, key=lambda f: f.stat().st_size)
        print(f"🎤 Selected reference: {best_file.name}")
        return best_file
    
    def test_with_f5tts_cli(self, text: str = "Hello, this is a test of voice cloning with F5-TTS."):
        """Test using F5-TTS command line interface"""
        ref_file = self.get_reference_sample()
        output_file = self.output_dir / "cloned_voice.wav"
        
        print(f"🎬 Generating speech with F5-TTS CLI...")
        print(f"   Text: '{text}'")
        print(f"   Reference: {ref_file.name}")
        
        try:
            # Use F5-TTS command line
            cmd = [
                "f5-tts_infer-cli",
                "--model", "F5-TTS",
                "--ref_audio", str(ref_file),
                "--ref_text", "This is a voice sample for training the speech synthesis model.",
                "--gen_text", text,
                "--output_path", str(output_file)
            ]
            
            print(f"🚀 Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Voice cloning successful!")
                print(f"💾 Output saved: {output_file}")
                return output_file
            else:
                print(f"❌ F5-TTS CLI failed:")
                print(f"   Error: {result.stderr}")
                return None
                
        except FileNotFoundError:
            print("❌ f5-tts_infer-cli not found")
            print("🔧 Trying alternative approach...")
            return self.test_with_python_api(text)
        except Exception as e:
            print(f"❌ CLI error: {e}")
            return None
    
    def test_with_python_api(self, text: str):
        """Test using Python API directly"""
        try:
            from f5_tts.api import F5TTS
            
            print("🔧 Using Python API...")
            ref_file = self.get_reference_sample()
            
            # Load model
            model = F5TTS()
            
            # Simple inference
            output_file = self.output_dir / "cloned_voice_api.wav"
            
            # This is a simplified approach - actual API may vary
            print(f"🎬 Generating with Python API...")
            
            # Note: This is a placeholder for the actual API call
            # The exact method depends on the F5-TTS version
            print("⚠️  Python API method needs F5-TTS specific implementation")
            print(f"📚 Check F5-TTS documentation for the exact API usage")
            
            return None
            
        except Exception as e:
            print(f"❌ Python API failed: {e}")
            return None
    
    def create_training_config(self):
        """Create configuration for fine-tuning"""
        training_dir = Path("../finetune_data/left_speaker_training")
        
        if not training_dir.exists():
            print(f"❌ Training data not found: {training_dir}")
            return None
        
        config = {
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
            "dataset_name": "Emilia_ZH_EN",
            "tokenizer": "pinyin",
            "data_path": str(training_dir),
            "notes": "Fine-tuning with left_speaker voice samples"
        }
        
        config_file = self.output_dir / "finetune_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"⚙️  Fine-tuning config created: {config_file}")
        print(f"📚 To start training:")
        print(f"   cd {training_dir.parent}")
        print(f"   f5-tts_train-cli --config {config_file}")
        
        return config_file
    
    def run_tests(self):
        """Run all available tests"""
        test_texts = [
            "Hello, this is a test of the voice cloning system.",
            "The weather is beautiful today.",
            "I hope this voice sounds natural and clear."
        ]
        
        print(f"🧪 Running voice cloning tests...")
        
        # Test 1: CLI approach
        print(f"\n🔬 Test 1: F5-TTS CLI")
        result1 = self.test_with_f5tts_cli(test_texts[0])
        
        # Test 2: Create training config
        print(f"\n🔬 Test 2: Training Configuration")
        config = self.create_training_config()
        
        # Summary
        print(f"\n📊 Test Summary:")
        print(f"   CLI test: {'✅ Success' if result1 else '❌ Failed'}")
        print(f"   Config created: {'✅ Yes' if config else '❌ No'}")
        
        if result1:
            print(f"\n🎉 Voice cloning test successful!")
            print(f"📁 Check outputs in: {self.output_dir}")
        else:
            print(f"\n⚠️  Voice cloning needs manual setup")
            print(f"📚 Refer to F5-TTS documentation for exact usage")

def main():
    samples_dir = "/Users/adamleeperelman/Documents/LLM/Voice Clone/ai_voice_samples/left_speaker"
    
    if not Path(samples_dir).exists():
        print(f"❌ Samples directory not found: {samples_dir}")
        return 1
    
    # Run tests
    cloner = SimpleVoiceCloner(samples_dir)
    cloner.run_tests()
    
    return 0

if __name__ == "__main__":
    exit(main())
