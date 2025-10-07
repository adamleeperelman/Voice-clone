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
            result = subprocess.run([sys.executable, '-c', 'import f5_tts; print("available")'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False
    
    def _find_finetuned_checkpoint(self) -> Optional[str]:
        """Find the most recent fine-tuned checkpoint in the workspace"""
        # Search common checkpoint locations
        search_paths = [
            Path(self.workspace_path) / "06_models" / "f5_tts",
            Path(self.workspace_path) / "F5_TTS" / "finetune_data" / "checkpoints",
            Path(self.workspace_path) / "checkpoints"
        ]
        
        checkpoint_files = []
        for search_path in search_paths:
            if search_path.exists():
                checkpoint_files.extend(search_path.glob("*.pt"))
                checkpoint_files.extend(search_path.glob("*.safetensors"))
        
        if not checkpoint_files:
            return None
        
        # Return the most recent checkpoint
        latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        return str(latest_checkpoint)
    
    def _get_reference_text(self, reference_audio: str) -> Optional[str]:
        """Extract reference text from training metadata if available"""
        try:
            # Get the filename from the reference audio path
            ref_filename = Path(reference_audio).name
            
            # Look for metadata in common locations
            metadata_paths = [
                Path(self.workspace_path) / "05_training_data" / "metadata.json",
                Path(self.workspace_path) / "F5_TTS" / "finetune_data" / "metadata.json"
            ]
            
            for metadata_path in metadata_paths:
                if not metadata_path.exists():
                    continue
                    
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Search for matching audio file in samples
                for sample in metadata.get('samples', []):
                    audio_path = sample.get('audio_path', '')
                    # Match by filename
                    if Path(audio_path).name == ref_filename or ref_filename in audio_path:
                        return sample.get('text', '')
            
            return None
        except Exception as e:
            return None
    
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
                      remove_silence: bool = True,
                      checkpoint_path: str = None,
                      nfe_step: int = 40,  # Increased from 32 to 40 for smoother output
                      cfg_strength: float = 1.5,  # Reduced from 2.0 to 1.5 to reduce artifacts
                      sway_sampling_coef: float = -1.0,
                      reference_text: str = None) -> str:
        """
        Generate audio using F5-TTS
        
        Args:
            text: Text to synthesize
            reference_audio: Path to reference audio file
            model: Model name (F5-TTS, E2-TTS, etc.)
            output_path: Output file path
            speed: Speech speed (default 1.0)
            remove_silence: Remove silence from output
            checkpoint_path: Path to fine-tuned model checkpoint (.pt file)
            nfe_step: Number of denoising steps (40 for smoother, less robotic output)
            cfg_strength: Classifier-free guidance strength (1.5 reduces artifacts)
            sway_sampling_coef: Sway sampling coefficient
            reference_text: Transcript of reference audio (helps with clarity)
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
        
        # Auto-detect fine-tuned checkpoint if not provided
        if not checkpoint_path:
            checkpoint_path = self._find_finetuned_checkpoint()
        
        # Auto-load reference text from metadata if not provided
        if not reference_text and reference_audio:
            reference_text = self._get_reference_text(reference_audio)
        
        print(f"🎤 Generating audio with F5-TTS...")
        print(f"   📝 Text: {text[:100]}{'...' if len(text) > 100 else ''}\"")
        print(f"   🎯 Reference: {Path(reference_audio).name}")
        if reference_text:
            print(f"   📄 Ref Text: {reference_text[:80]}{'...' if len(reference_text) > 80 else ''}\"")
        print(f"   🤖 Model: {model}")
        if checkpoint_path:
            print(f"   🔬 Checkpoint: {Path(checkpoint_path).name}")
        print(f"   ⚙️  NFE Steps: {nfe_step} (quality)")
        print(f"   📊 CFG Strength: {cfg_strength}")
        
        try:
            # Map model names to F5-TTS CLI expected format
            model_map = {
                "F5-TTS": "F5TTS_Base",
                "E2-TTS": "E2TTS_Base",
                "F5TTS": "F5TTS_Base",
                "E2TTS": "E2TTS_Base"
            }
            cli_model = model_map.get(model, model)
            
            # Construct F5-TTS command
            cmd = [
                sys.executable, "-m", "f5_tts.infer.infer_cli",
                "--model", cli_model,
                "--ref_audio", reference_audio,
                "--gen_text", text,
                "--output_dir", str(Path(output_path).parent),
                "--output_file", Path(output_path).name,
                "--nfe_step", str(nfe_step),
                "--cfg_strength", str(cfg_strength),
                "--sway_sampling_coef", str(sway_sampling_coef)
            ]
            
            # Add reference text if provided (helps with intelligibility)
            if reference_text:
                cmd.extend(["--ref_text", reference_text])
            
            # Add fine-tuned checkpoint if available
            if checkpoint_path and os.path.exists(checkpoint_path):
                cmd.extend(["--ckpt_file", checkpoint_path])
            
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
                      name_prefix: str = "generated",
                      checkpoint_path: str = None,
                      **kwargs) -> List[str]:
        """
        Generate multiple audio files in batch
        """
        print(f"🎯 Starting batch audio generation ({len(texts)} items)...")
        
        # Auto-detect checkpoint if not provided
        if not checkpoint_path:
            checkpoint_path = self._find_finetuned_checkpoint()
            if checkpoint_path:
                print(f"🔬 Using fine-tuned checkpoint: {Path(checkpoint_path).name}")
        
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
                output_path=str(file_path),
                checkpoint_path=checkpoint_path,
                **kwargs
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
                          output_dir: str = "voice_tests",
                          checkpoint_path: str = None,
                          **kwargs) -> List[str]:
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
        
        # Auto-detect checkpoint if not provided
        if not checkpoint_path:
            checkpoint_path = self._find_finetuned_checkpoint()
            if checkpoint_path:
                print(f"🔬 Using fine-tuned checkpoint: {Path(checkpoint_path).name}")
        
        # Create test output directory
        test_output_path = Path(self.workspace_path) / output_dir
        test_output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        for i, (phrase, name) in enumerate(zip(test_phrases, test_names), 1):
            print(f"\n🎯 Test {i}/4: {name}")
            
            filename = f"test_{i}_{name}.wav"
            file_path = test_output_path / filename
            
            result_path = self.generate_audio(
                text=phrase,
                reference_audio=reference_audio,
                model=model,
                output_path=str(file_path),
                checkpoint_path=checkpoint_path,
                **kwargs
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
