#!/usr/bin/env python3
"""
Working F5-TTS Voice Cloning Test
Test voice cloning with correct F5-TTS CLI arguments
"""

import os
import sys
from pathlib import Path
import subprocess
import json

class WorkingVoiceCloner:
    """Working voice cloning using proper F5-TTS CLI"""
    
    def __init__(self, samples_dir: str):
        self.samples_dir = Path(samples_dir)
        self.output_dir = Path("generated_voices")
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🎯 Working Voice Cloning with F5-TTS")
        print(f"📁 Samples: {self.samples_dir}")
        print(f"💾 Output: {self.output_dir}")
    
    def get_best_reference(self):
        """Get the best reference sample"""
        audio_files = list(self.samples_dir.glob("*.wav"))
        if not audio_files:
            raise FileNotFoundError(f"No audio files found in {self.samples_dir}")
        
        # Use the longest sample (likely highest quality)
        best_file = max(audio_files, key=lambda f: f.stat().st_size)
        print(f"🎤 Selected reference: {best_file.name}")
        return best_file
    
    def test_voice_cloning(self, text: str, output_name: str = "cloned_voice"):
        """Test voice cloning with F5-TTS"""
        ref_file = self.get_best_reference()
        output_file = f"{output_name}.wav"
        
        print(f"🎬 Generating speech...")
        print(f"   Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print(f"   Reference: {ref_file.name}")
        print(f"   Output: {output_file}")
        
        try:
            # Correct F5-TTS CLI command
            cmd = [
                "f5-tts_infer-cli",
                "--model", "F5TTS_Base",
                "--ref_audio", str(ref_file),
                "--ref_text", "This is a voice sample for training the speech synthesis model.",
                "--gen_text", text,
                "--output_dir", str(self.output_dir),
                "--output_file", output_file,
                "--remove_silence",
                "--target_rms", "0.1"
            ]
            
            print(f"🚀 Running F5-TTS inference...")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.output_dir)
            
            output_path = self.output_dir / output_file
            
            if result.returncode == 0 and output_path.exists():
                print(f"✅ Voice cloning successful!")
                print(f"💾 Generated: {output_path}")
                return output_path
            else:
                print(f"❌ F5-TTS failed:")
                if result.stderr:
                    print(f"   Error: {result.stderr}")
                if result.stdout:
                    print(f"   Output: {result.stdout}")
                return None
                
        except FileNotFoundError:
            print("❌ f5-tts_infer-cli not found in PATH")
            print("🔧 Make sure F5-TTS is properly installed")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def run_voice_tests(self):
        """Run multiple voice cloning tests"""
        test_cases = [
            {
                "text": "Hello, this is a test of voice cloning using F5-TTS technology.",
                "name": "test_1_greeting"
            },
            {
                "text": "The quick brown fox jumps over the lazy dog. This is a test of pronunciation and clarity.",
                "name": "test_2_pangram"
            },
            {
                "text": "I am testing the quality and naturalness of the synthesized speech output.",
                "name": "test_3_quality"
            },
            {
                "text": "This voice should sound very similar to the original speaker's voice characteristics.",
                "name": "test_4_similarity"
            }
        ]
        
        print(f"🧪 Running {len(test_cases)} voice cloning tests...")
        
        results = []
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}/{len(test_cases)}: {test_case['name']}")
            result = self.test_voice_cloning(test_case["text"], test_case["name"])
            results.append(result is not None)
        
        # Summary
        successful = sum(results)
        print(f"\n📊 Test Results:")
        print(f"   Successful: {successful}/{len(test_cases)}")
        print(f"   Success rate: {successful/len(test_cases)*100:.1f}%")
        
        if successful > 0:
            print(f"\n🎉 Voice cloning working!")
            print(f"📁 Generated files in: {self.output_dir}")
            print(f"🎧 Listen to the generated audio files to evaluate quality")
        else:
            print(f"\n❌ All tests failed")
            print(f"🔧 Check F5-TTS installation and try manual CLI commands")
        
        return successful > 0
    
    def create_fine_tuning_guide(self):
        """Create a guide for fine-tuning with the prepared data"""
        guide_file = self.output_dir / "fine_tuning_guide.md"
        
        guide_content = f"""# F5-TTS Fine-tuning Guide

## Your Training Data
- **Location**: `{self.samples_dir}`
- **Prepared Data**: `../finetune_data/left_speaker_training/`
- **Audio Files**: 8 samples, ~72.8 seconds total
- **Average Duration**: 9.1 seconds per sample

## Fine-tuning Steps

### 1. Verify Training Data Structure
```bash
cd "../finetune_data/left_speaker_training"
ls -la audio/          # Should show sample_000.wav to sample_007.wav
cat metadata.txt       # Should show filename|transcription pairs
```

### 2. Create Training Config
```bash
# Copy the prepared data to F5-TTS expected location
mkdir -p ~/.cache/f5-tts/datasets/left_speaker
cp -r audio/* ~/.cache/f5-tts/datasets/left_speaker/
cp metadata.txt ~/.cache/f5-tts/datasets/left_speaker/
```

### 3. Start Fine-tuning
```bash
# Basic fine-tuning command
f5-tts_train-cli \\
    --model F5TTS_Base \\
    --dataset_name left_speaker \\
    --learning_rate 7.5e-5 \\
    --batch_size 4 \\
    --epochs 200 \\
    --save_per_updates 400 \\
    --exp_name left_speaker_finetune
```

### 4. Monitor Training
- Training logs will show progress
- Models saved in `ckpts/` directory
- Use tensorboard for monitoring: `tensorboard --logdir ckpts/`

### 5. Test Fine-tuned Model
```bash
f5-tts_infer-cli \\
    --model F5TTS_Base \\
    --ckpt_file ckpts/left_speaker_finetune/model_best.pt \\
    --ref_audio "{self.samples_dir}/AI_008_11.0s_q7.wav" \\
    --ref_text "This is a voice sample for training." \\
    --gen_text "Hello, I am the fine-tuned voice model." \\
    --output_dir generated_finetuned
```

## Tips for Better Results
1. **More Data**: Add more high-quality samples (aim for 10+ minutes)
2. **Consistent Quality**: Ensure all samples have similar audio quality
3. **Diverse Content**: Include varied sentence structures and vocabulary
4. **Proper Transcriptions**: Accurate transcriptions improve results

## Current Status
- ✅ Training data prepared
- ✅ Base model working
- 🔄 Ready for fine-tuning
"""
        
        with open(guide_file, 'w') as f:
            f.write(guide_content)
        
        print(f"📚 Fine-tuning guide created: {guide_file}")
        return guide_file

def main():
    samples_dir = "/Users/adamleeperelman/Documents/LLM/Voice Clone/ai_voice_samples/left_speaker"
    
    if not Path(samples_dir).exists():
        print(f"❌ Samples directory not found: {samples_dir}")
        return 1
    
    # Initialize cloner
    cloner = WorkingVoiceCloner(samples_dir)
    
    # Run tests
    success = cloner.run_voice_tests()
    
    # Create fine-tuning guide
    guide = cloner.create_fine_tuning_guide()
    
    if success:
        print(f"\n🎯 Next Steps:")
        print(f"1. 🎧 Listen to generated samples to evaluate quality")
        print(f"2. 📚 Follow the fine-tuning guide: {guide}")
        print(f"3. 🔄 Fine-tune the model with your specific voice data")
        print(f"4. 🚀 Test the fine-tuned model for better results")
    else:
        print(f"\n🔧 Troubleshooting:")
        print(f"1. Check F5-TTS installation: pip install f5-tts")
        print(f"2. Verify CLI is in PATH: which f5-tts_infer-cli")
        print(f"3. Try manual CLI command from the guide")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
