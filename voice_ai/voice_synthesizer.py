#!/usr/bin/env python3
"""
Voice Synthesizer Module
Handles F5-TTS voice synthesis and generation
"""

import os
import subprocess
import sys
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Optional, Union
import warnings
warnings.filterwarnings("ignore")

class VoiceSynthesizer:
    """
    Voice Synthesis Module for F5-TTS
    Handles voice generation using pre-trained or fine-tuned models
    """
    
    def __init__(self, workspace_path: str = None):
        """Initialize the Voice Synthesizer"""
        self.workspace_path = workspace_path or os.getcwd()
        self.f5_tts_available = self._check_f5_tts()
        print("🎤 Voice Synthesizer initialized")
    
    def _check_f5_tts(self) -> bool:
        """Check if F5-TTS is available"""
        try:
            result = subprocess.run(['python', '-c', 'import f5_tts; print("available")'], 
                                  capture_output=True, text=True, cwd=self.workspace_path)
            return result.returncode == 0
        except Exception:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available F5-TTS models"""
        # Default F5-TTS models
        default_models = [
            "F5-TTS",
            "E2-TTS", 
            "custom"  # For fine-tuned models
        ]
        
        # Check for custom models in finetune directory
        custom_models = []
        finetune_path = Path(self.workspace_path) / "F5_TTS" / "finetune_data"
        if finetune_path.exists():
            checkpoints_dir = finetune_path / "checkpoints"
            if checkpoints_dir.exists():
                for checkpoint in checkpoints_dir.glob("*.pt"):
                    custom_models.append(f"custom_{checkpoint.stem}")
        
        return default_models + custom_models
    
    def find_reference_audio(self, speaker_dir: str = None) -> Optional[str]:
        """Find suitable reference audio for voice cloning"""
        if not speaker_dir:
            # Default to left_speaker samples
            speaker_dir = os.path.join(self.workspace_path, "ai_voice_samples", "left_speaker")
        
        if not os.path.exists(speaker_dir):
            print(f"⚠️  Speaker directory not found: {speaker_dir}")
            return None
        
        # Look for audio files
        audio_extensions = {'.wav', '.mp3', '.m4a', '.flac'}
        audio_files = []
        
        for file_path in Path(speaker_dir).glob('*'):
            if file_path.suffix.lower() in audio_extensions:
                audio_files.append(str(file_path))
        
        if not audio_files:
            print(f"⚠️  No audio files found in {speaker_dir}")
            return None
        
        # Return the first audio file (or could implement selection logic)
        selected_file = audio_files[0]
        print(f"🎯 Using reference audio: {Path(selected_file).name}")
        return selected_file
    
    def generate_audio(self, 
                      text: str, 
                      reference_audio: str = None, 
                      model: str = "F5-TTS", 
                      output_path: str = None,
                      speed: float = 1.0,
                      remove_silence: bool = True) -> str:
        """
        Generate audio using F5-TTS
        """
        if not self.f5_tts_available:
            print("❌ F5-TTS not available. Install with: pip install f5-tts")
            return ""
        
        # Find reference audio if not provided
        if not reference_audio:
            reference_audio = self.find_reference_audio()
            if not reference_audio:
                print("❌ No reference audio found")
                return ""
        
        # Generate output path if not provided
        if not output_path:
            output_dir = os.path.join(self.workspace_path, "generated_audio")
            os.makedirs(output_dir, exist_ok=True)
            
            # Create safe filename from text
            safe_text = "".join(c for c in text[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_text = safe_text.replace(' ', '_')
            output_path = os.path.join(output_dir, f"generated_{safe_text}.wav")
        
        # Ensure reference audio exists
        if not os.path.exists(reference_audio):
            print(f"❌ Reference audio not found: {reference_audio}")
            return ""
        
        print(f"🎤 Generating audio with F5-TTS...")
        print(f"   📝 Text: {text[:100]}{'...' if len(text) > 100 else ''}\"")
        print(f"   🎯 Reference: {Path(reference_audio).name}")
        print(f"   🤖 Model: {model}")
        
        try:
            # Construct F5-TTS command
            cmd = [
                "f5-tts_infer-cli",
                "--model", model,
                "--ref_audio", reference_audio,
                "--gen_text", text,
                "--output_dir", str(Path(output_path).parent),
                "--output_file", Path(output_path).name
            ]
            
            # Add optional parameters
            if speed != 1.0:
                cmd.extend(["--speed", str(speed)])
            
            if remove_silence:
                cmd.append("--remove_silence")
            
            print(f"🚀 Running F5-TTS inference...")
            
            # Run the command
            result = subprocess.run(
                cmd,
                cwd=self.workspace_path,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Check if output file was created
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"✅ Audio generated successfully!")
                print(f"   📁 Output: {output_path}")
                print(f"   📊 Size: {file_size / 1024:.1f} KB")
                return output_path
            else:
                print("❌ Audio generation failed - no output file created")
                if result.stdout:
                    print(f"   stdout: {result.stdout}")
                if result.stderr:
                    print(f"   stderr: {result.stderr}")
                return ""
                
        except subprocess.TimeoutExpired:
            print("❌ Audio generation timed out (5 minutes)")
            return ""
        except Exception as e:
            print(f"❌ Error during audio generation: {e}")
            return ""
    
    def batch_generate(self, 
                      texts: List[str], 
                      reference_audio: str = None, 
                      model: str = "F5-TTS",
                      output_dir: str = "generated_audio",
                      name_prefix: str = "generated") -> List[str]:
        """
        Generate multiple audio files in batch
        """
        print(f"🎯 Starting batch audio generation ({len(texts)} items)...")
        
        # Create output directory
        output_path = Path(self.workspace_path) / output_dir
        output_path.mkdir(exist_ok=True)
        
        generated_files = []
        
        for i, text in enumerate(texts, 1):
            print(f"\n📝 Generating {i}/{len(texts)}...")
            
            # Create unique output filename
            safe_text = "".join(c for c in text[:20] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_text = safe_text.replace(' ', '_')
            filename = f"{name_prefix}_{i:03d}_{safe_text}.wav"
            file_path = output_path / filename
            
            # Generate audio
            result_path = self.generate_audio(
                text=text,
                reference_audio=reference_audio,
                model=model,
                output_path=str(file_path)
            )
            
            if result_path:
                generated_files.append(result_path)
                print(f"   ✅ Generated: {filename}")
            else:
                print(f"   ❌ Failed: {filename}")
        
        print(f"\n🎉 Batch generation complete!")
        print(f"   ✅ Successfully generated: {len(generated_files)}/{len(texts)} files")
        print(f"   📁 Output directory: {output_path}")
        
        return generated_files
    
    def test_voice_cloning(self, 
                          reference_audio: str = None, 
                          model: str = "F5-TTS",
                          output_dir: str = "voice_tests") -> List[str]:
        """
        Run voice cloning tests with predefined test phrases
        """
        test_phrases = [
            "Hello, this is a test of the voice cloning system.",
            "The quick brown fox jumps over the lazy dog.",
            "Voice synthesis quality depends on the reference audio and model training.",
            "This generated speech should sound similar to the original speaker."
        ]
        
        test_names = ["greeting", "pangram", "quality", "similarity"]
        
        print(f"🧪 Running voice cloning tests...")
        
        # Create test output directory
        test_output_path = Path(self.workspace_path) / output_dir
        test_output_path.mkdir(exist_ok=True)
        
        generated_files = []
        
        for i, (phrase, name) in enumerate(zip(test_phrases, test_names), 1):
            print(f"\n🎯 Test {i}/4: {name}")
            
            filename = f"test_{i}_{name}.wav"
            file_path = test_output_path / filename
            
            result_path = self.generate_audio(
                text=phrase,
                reference_audio=reference_audio,
                model=model,
                output_path=str(file_path)
            )
            
            if result_path:
                generated_files.append(result_path)
                print(f"   ✅ Test complete: {filename}")
            else:
                print(f"   ❌ Test failed: {filename}")
        
        print(f"\n🧪 Voice cloning tests complete!")
        print(f"   ✅ Successful tests: {len(generated_files)}/4")
        print(f"   📁 Test output: {test_output_path}")
        
        return generated_files
    
    def synthesize_from_file(self, 
                           text_file: str, 
                           reference_audio: str = None,
                           model: str = "F5-TTS",
                           output_dir: str = "synthesized_audio") -> List[str]:
        """
        Synthesize audio from text file (one paragraph per audio file)
        """
        if not os.path.exists(text_file):
            print(f"❌ Text file not found: {text_file}")
            return []
        
        # Read text file
        with open(text_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if not paragraphs:
            print("❌ No paragraphs found in text file")
            return []
        
        print(f"📖 Found {len(paragraphs)} paragraphs to synthesize")
        
        # Generate audio for each paragraph
        return self.batch_generate(
            texts=paragraphs,
            reference_audio=reference_audio,
            model=model,
            output_dir=output_dir,
            name_prefix="paragraph"
        )
    
    def get_synthesis_status(self) -> Dict:
        """Get status of voice synthesis capabilities"""
        status = {
            "f5_tts_available": self.f5_tts_available,
            "available_models": self.get_available_models(),
            "reference_audio_found": False,
            "generated_audio_count": 0
        }
        
        # Check for reference audio
        reference_path = self.find_reference_audio()
        if reference_path:
            status["reference_audio_found"] = True
            status["reference_audio_path"] = reference_path
        
        # Count generated audio files
        generated_dir = Path(self.workspace_path) / "generated_audio"
        if generated_dir.exists():
            audio_files = list(generated_dir.glob("*.wav"))
            status["generated_audio_count"] = len(audio_files)
        
        return status
