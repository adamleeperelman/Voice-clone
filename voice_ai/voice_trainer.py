#!/usr/bin/env python3
"""
Voice Trainer Module
Handles F5-TTS training data preparation and fine-tuning
"""

import os
import json
import re
import shutil
import subprocess
import sys
from importlib.resources import files
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

try:
    from datasets import Dataset
except ImportError:  # pragma: no cover - datasets should ship with f5-tts
    Dataset = None

# Import whisper safely
def load_whisper():
    try:
        import whisper
        return whisper
    except Exception as e:
        print(f"❌ Error loading Whisper: {e}")
        print("Install with: pip install openai-whisper")
        sys.exit(1)

class VoiceTrainer:
    """
    Voice Training Module for F5-TTS
    Handles training data preparation and fine-tuning
    """
    
    def __init__(self, workspace_path: str = None):
        """Initialize the Voice Trainer"""
        self.workspace_path = workspace_path or os.getcwd()
        self.whisper_model = None
        self._f5_data_root: Optional[Path] = None
        print("🎯 Voice Trainer initialized")
    
    def load_whisper_model(self, model_size: str = "base") -> object:
        """Load Whisper model for transcription"""
        if self.whisper_model is None:
            print(f"🔥 Loading Whisper model ({model_size})...")
            whisper = load_whisper()
            self.whisper_model = whisper.load_model(model_size)
            print("✅ Whisper model loaded")
        return self.whisper_model
    
    def scan_audio_files(self, directory: str) -> List[str]:
        """Scan directory for audio files"""
        audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
        audio_files = []
        
        for file_path in Path(directory).rglob('*'):
            if file_path.suffix.lower() in audio_extensions:
                audio_files.append(str(file_path))
        
        return sorted(audio_files)

    # ------------------------------------------------------------------
    # F5-TTS dataset preparation helpers
    # ------------------------------------------------------------------

    def _resolve_f5_data_root(self) -> Path:
        """Locate the data directory used by F5-TTS for custom datasets."""

        if self._f5_data_root and self._f5_data_root.exists():
            return self._f5_data_root

        try:
            base_path = Path(files("f5_tts").joinpath("../../data")).resolve()
        except Exception:
            base_path = Path(self.workspace_path) / "F5_TTS" / "data"

        base_path.mkdir(parents=True, exist_ok=True)
        self._f5_data_root = base_path
        return self._f5_data_root

    def _normalise_dataset_name(self, raw_name: str) -> str:
        """Convert an arbitrary string into a filesystem-safe dataset identifier."""

        cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", raw_name.strip()) or "custom_dataset"
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        return cleaned.lower() or "custom_dataset"

    def prepare_f5_dataset(
        self,
        training_dir: str,
        dataset_name: str,
        tokenizer: str = "byte",
        refresh: bool = True,
    ) -> Dict[str, str]:
        """Export training data into the HuggingFace dataset format expected by F5-TTS."""

        if Dataset is None:
            raise RuntimeError(
                "🤖 The 'datasets' package is required to build F5 datasets. Install it with `pip install datasets`."
            )

        training_path = Path(training_dir)
        if not training_path.exists():
            candidate = Path(self.workspace_path) / training_dir
            if candidate.exists():
                training_path = candidate
        metadata_path = training_path / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Training metadata not found at {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        samples = metadata.get("samples", [])
        if not samples:
            raise ValueError("Training metadata contains no samples to export")

        dataset_entries: List[Dict[str, object]] = []
        durations: List[float] = []
        skipped = 0

        for sample in samples:
            audio_rel = sample.get("audio_path")
            text = sample.get("text", "") or ""
            duration = float(sample.get("duration", 0.0))

            if not audio_rel:
                skipped += 1
                continue

            audio_path = (training_path / audio_rel).resolve()
            if not audio_path.exists():
                skipped += 1
                continue

            dataset_entries.append(
                {
                    "audio_path": str(audio_path),
                    "text": text,
                    "duration": duration,
                }
            )
            durations.append(duration)

        if not dataset_entries:
            raise ValueError("No training samples with valid audio paths were found")

        if skipped:
            print(f"⚠️  Skipped {skipped} samples without valid audio")

        dataset_id = self._normalise_dataset_name(dataset_name)
        data_root = self._resolve_f5_data_root()
        dataset_dir = data_root / f"{dataset_id}_{tokenizer}"

        if refresh and dataset_dir.exists():
            shutil.rmtree(dataset_dir)

        raw_dir = dataset_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        print(f"🧱 Building F5 dataset: {dataset_dir}")
        ds = Dataset.from_list(dataset_entries)
        ds.save_to_disk(str(raw_dir))

        duration_payload = {"duration": durations}
        with open(dataset_dir / "duration.json", "w", encoding="utf-8") as f:
            json.dump(duration_payload, f, indent=2)

        manifest = {
            "dataset_name": dataset_id,
            "tokenizer": tokenizer,
            "source_metadata": str(metadata_path),
            "num_samples": len(dataset_entries),
            "skipped_samples": skipped,
        }
        with open(dataset_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(
            f"✅ F5 dataset ready: {dataset_dir} (samples={len(dataset_entries)}, tokenizer={tokenizer})"
        )

        return {
            "dataset_dir": str(dataset_dir),
            "dataset_name": dataset_id,
            "tokenizer": tokenizer,
            "num_samples": len(dataset_entries),
            "total_duration": float(sum(durations)),
        }
    
    def transcribe_audio_file(self, audio_path: str) -> str:
        """Transcribe a single audio file using Whisper"""
        if not self.whisper_model:
            self.load_whisper_model()
        
        try:
            result = self.whisper_model.transcribe(
                audio_path,
                language="en",
                task="transcribe",
                temperature=0.0,
                verbose=False
            )
            return result['text'].strip()
        except Exception as e:
            print(f"⚠️  Error transcribing {audio_path}: {e}")
            return ""
    
    def prepare_training_data(self, 
                            source_dir: str, 
                            output_dir: str = "F5_TTS/finetune_data",
                            speaker_name: str = "custom_speaker") -> Dict:
        """
        Prepare training data for F5-TTS from audio samples
        """
        print(f"🎯 Preparing F5-TTS training data from: {source_dir}")
        
        # Create output directories
        output_path = Path(self.workspace_path) / output_dir
        wavs_dir = output_path / "wavs"
        wavs_dir.mkdir(parents=True, exist_ok=True)
        
        # Scan for audio files
        audio_files = self.scan_audio_files(source_dir)
        if not audio_files:
            print(f"❌ No audio files found in {source_dir}")
            return {}
        
        print(f"📁 Found {len(audio_files)} audio files")
        
        # Process each audio file
        training_entries = []
        total_duration = 0
        processed_count = 0
        
        for i, audio_path in enumerate(audio_files):
            try:
                print(f"🎤 Processing {i+1}/{len(audio_files)}: {Path(audio_path).name}")
                
                # Get file info
                from pydub import AudioSegment
                audio = AudioSegment.from_file(audio_path)
                duration = len(audio) / 1000.0  # Convert to seconds
                
                # Skip files that are too short or too long
                if duration < 3.0 or duration > 30.0:
                    print(f"   ⚠️  Skipping {Path(audio_path).name}: duration {duration:.1f}s (should be 3-30s)")
                    continue
                
                # Transcribe the audio
                transcription = self.transcribe_audio_file(audio_path)
                if not transcription or len(transcription.split()) < 5:
                    print(f"   ⚠️  Skipping {Path(audio_path).name}: poor transcription")
                    continue
                
                # Copy/convert audio to training directory
                training_filename = f"{speaker_name}_{i+1:03d}.wav"
                training_path = wavs_dir / training_filename
                
                # Ensure audio is in correct format (16kHz, mono)
                audio = audio.set_frame_rate(16000).set_channels(1)
                audio.export(str(training_path), format="wav")
                
                # Create training entry
                training_entries.append({
                    "audio_path": str(training_path.relative_to(output_path)),
                    "text": transcription,
                    "speaker": speaker_name,
                    "duration": duration
                })
                
                total_duration += duration
                processed_count += 1
                
                print(f"   ✅ Processed: {duration:.1f}s - '{transcription[:50]}{'...' if len(transcription) > 50 else ''}'")
                
            except Exception as e:
                print(f"   ❌ Error processing {audio_path}: {e}")
                continue
        
        if not training_entries:
            print("❌ No valid training samples created")
            return {}
        
        # Create metadata file for F5-TTS
        metadata = {
            "speaker": speaker_name,
            "total_samples": len(training_entries),
            "total_duration": total_duration,
            "average_duration": total_duration / len(training_entries),
            "samples": training_entries
        }
        
        # Save metadata as JSON
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Create F5-TTS compatible file list
        filelist_path = output_path / "filelist.txt"
        with open(filelist_path, 'w', encoding='utf-8') as f:
            for entry in training_entries:
                f.write(f"{entry['audio_path']}|{entry['text']}\n")
        
        print(f"✅ Training data prepared:")
        print(f"   📁 Output directory: {output_path}")
        print(f"   🎵 Audio samples: {len(training_entries)}")
        print(f"   ⏱️  Total duration: {total_duration:.1f} seconds")
        print(f"   📄 Metadata: {metadata_path}")
        print(f"   📋 Filelist: {filelist_path}")
        
        return metadata
    
    def validate_training_data(self, training_dir: str = None) -> Dict:
        """Validate training data for F5-TTS compatibility"""
        if training_dir is None:
            # Auto-detect training directory by looking for common patterns
            possible_dirs = [
                "05_training_data",  # Pipeline generated
                "F5_TTS/finetune_data",  # Default F5-TTS location
                "training_data"  # Generic fallback
            ]
            training_path = None
            for dir_name in possible_dirs:
                candidate = Path(self.workspace_path) / dir_name
                if candidate.exists() and (candidate / "metadata.json").exists():
                    training_path = candidate
                    training_dir = dir_name
                    break
            
            if training_path is None:
                # Default to F5_TTS path even if it doesn't exist
                training_path = Path(self.workspace_path) / "F5_TTS/finetune_data"
                training_dir = "F5_TTS/finetune_data"
        else:
            training_path = Path(self.workspace_path) / training_dir
            
        print(f"🔍 Validating training data in: {training_dir}")
        
        # Check required files
        required_files = ["metadata.json", "filelist.txt"]
        missing_files = []
        
        for file in required_files:
            if not (training_path / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ Missing required files: {missing_files}")
            return {"valid": False, "errors": f"Missing files: {missing_files}"}
        
        # Check wavs directory
        wavs_dir = training_path / "wavs"
        if not wavs_dir.exists():
            print("❌ Missing wavs directory")
            return {"valid": False, "errors": "Missing wavs directory"}
        
        # Load and validate metadata
        try:
            with open(training_path / "metadata.json", 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"❌ Error reading metadata.json: {e}")
            return {"valid": False, "errors": f"Invalid metadata.json: {e}"}
        
        # Validate samples
        validation_results = {
            "valid": True,
            "total_samples": metadata.get("total_samples", 0),
            "total_duration": metadata.get("total_duration", 0),
            "issues": []
        }
        
        samples = metadata.get("samples", [])
        for i, sample in enumerate(samples):
            audio_path = training_path / sample["audio_path"]
            
            # Check if audio file exists
            if not audio_path.exists():
                validation_results["issues"].append(f"Sample {i+1}: Audio file missing - {audio_path}")
                continue
            
            # Check text quality
            text = sample.get("text", "")
            if not text or len(text.split()) < 3:
                validation_results["issues"].append(f"Sample {i+1}: Text too short - '{text}'")
            
            # Check duration
            duration = sample.get("duration", 0)
            if duration < 3.0 or duration > 30.0:
                validation_results["issues"].append(f"Sample {i+1}: Duration {duration:.1f}s outside recommended range (3-30s)")
        
        if validation_results["issues"]:
            print(f"⚠️  Found {len(validation_results['issues'])} validation issues:")
            for issue in validation_results["issues"][:5]:  # Show first 5 issues
                print(f"   - {issue}")
            if len(validation_results["issues"]) > 5:
                print(f"   ... and {len(validation_results['issues']) - 5} more issues")
        else:
            print("✅ Training data validation passed")
        
        print(f"📊 Training data summary:")
        print(f"   🎵 Total samples: {validation_results['total_samples']}")
        print(f"   ⏱️  Total duration: {validation_results['total_duration']:.1f}s")
        print(f"   ⚠️  Issues found: {len(validation_results['issues'])}")
        
        return validation_results
    
    def fine_tune_model(self, 
                       training_dir: str = "F5_TTS/finetune_data",
                       model_name: str = "custom_voice",
                       epochs: int = 50,
                       learning_rate: float = 1e-4,
                       batch_size: int = 8) -> Dict:
        """
        Fine-tune F5-TTS model with prepared training data
        """
        print(f"🚀 Starting F5-TTS fine-tuning...")
        
        # Validate training data first
        validation = self.validate_training_data(training_dir)
        if not validation.get("valid", True) or validation.get("issues", []):
            if validation.get("issues", []):
                print(f"⚠️  Training data has {len(validation['issues'])} issues but proceeding...")
            else:
                print("❌ Training data validation failed")
                return {"success": False, "error": "Training data validation failed"}
        
        training_path = Path(self.workspace_path) / training_dir
        
        # Check if F5-TTS is installed and accessible
        try:
            result = subprocess.run(['python', '-c', 'import f5_tts; print("F5-TTS available")'], 
                                  capture_output=True, text=True, cwd=self.workspace_path)
            if result.returncode != 0:
                print("❌ F5-TTS not found. Install with: pip install f5-tts")
                return {"success": False, "error": "F5-TTS not installed"}
        except Exception as e:
            print(f"❌ Error checking F5-TTS: {e}")
            return {"success": False, "error": f"F5-TTS check failed: {e}"}
        
        # Prepare fine-tuning command
        # Note: This is a template - actual F5-TTS fine-tuning commands may vary
        finetune_script = f"""
import os
import sys
sys.path.append('{self.workspace_path}')

# Import F5-TTS fine-tuning modules
try:
    from f5_tts.train import train_model
    from f5_tts.configs import TrainingConfig
    
    # Configure training
    config = TrainingConfig(
        data_dir='{training_path}',
        model_name='{model_name}',
        epochs={epochs},
        learning_rate={learning_rate},
        batch_size={batch_size},
        save_dir='{training_path}/checkpoints'
    )
    
    # Start training
    print("🚀 Starting F5-TTS fine-tuning...")
    train_model(config)
    print("✅ Fine-tuning completed!")
    
except ImportError as e:
    print(f"❌ F5-TTS training modules not found: {{e}}")
    print("This is a template - actual F5-TTS fine-tuning may require different setup")
    
except Exception as e:
    print(f"❌ Fine-tuning error: {{e}}")
"""
        
        # Save fine-tuning script
        script_path = training_path / "finetune_script.py"
        with open(script_path, 'w') as f:
            f.write(finetune_script)
        
        print(f"📝 Fine-tuning script saved: {script_path}")
        print(f"⚠️  Note: F5-TTS fine-tuning implementation may require specific setup")
        print(f"📚 Refer to F5-TTS documentation for exact fine-tuning procedures")
        
        # For now, return configuration info
        return {
            "success": True,
            "message": "Fine-tuning configuration prepared",
            "script_path": str(script_path),
            "training_dir": str(training_path),
            "model_name": model_name,
            "config": {
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size
            },
            "note": "Actual fine-tuning requires F5-TTS specific implementation"
        }
    
    def get_training_status(self, training_dir: str = None) -> Dict:
        """Get status of training data and fine-tuning progress"""
        if training_dir is None:
            # Auto-detect training directory by looking for common patterns
            possible_dirs = [
                "05_training_data",  # Pipeline generated
                "F5_TTS/finetune_data",  # Default F5-TTS location
                "training_data"  # Generic fallback
            ]
            training_path = None
            for dir_name in possible_dirs:
                candidate = Path(self.workspace_path) / dir_name
                if candidate.exists() and (candidate / "metadata.json").exists():
                    training_path = candidate
                    break
            
            if training_path is None:
                # Default to F5_TTS path even if it doesn't exist
                training_path = Path(self.workspace_path) / "F5_TTS/finetune_data"
        else:
            training_path = Path(self.workspace_path) / training_dir
        
        status = {
            "training_dir_exists": training_path.exists(),
            "training_data_prepared": False,
            "total_samples": 0,
            "total_duration": 0,
            "validation_status": {},
            "checkpoints_exist": False
        }
        
        # Check training data
        if training_path.exists():
            metadata_path = training_path / "metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    status["training_data_prepared"] = True
                    status["total_samples"] = metadata.get("total_samples", 0)
                    status["total_duration"] = metadata.get("total_duration", 0)
                except Exception:
                    pass
            
            # Check validation
            try:
                status["validation_status"] = self.validate_training_data(training_dir)
            except Exception:
                pass
            
            # Check for checkpoints
            checkpoints_dir = training_path / "checkpoints"
            if checkpoints_dir.exists() and any(checkpoints_dir.iterdir()):
                status["checkpoints_exist"] = True
        
        return status
