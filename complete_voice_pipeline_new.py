#!/usr/bin/env python3
#%%
"""
Complete Voice Cloning Pipeline
End-to-end script using the modular Voice AI system
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union
from importlib.resources import files

# Import the voice AI modules
from voice_ai import create_processor

def create_project_structure(project_name: str = "voice_clone_project") -> Path:
    """Create a new project structure for the voice cloning pipeline"""
    
    # Create project directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_dir = Path(f"{project_name}_{timestamp}").resolve()
    
    # Create subdirectories
    subdirs = [
        "01_extracted_audio",      # Time-extracted audio
        "02_stereo_channels",      # Left and right channel separation
        "03_separated_voices",     # Speaker-separated audio
        "04_voice_filtered",       # Voice activity filtered audio
        "05_training_data",        # Prepared training data
        "06_models",              # Fine-tuned models
        "07_generated_audio",     # Synthesized output
        "logs"                    # Process logs
    ]
    
    for subdir in subdirs:
        (project_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Created project structure: {project_dir}")
    return project_dir

def log_step(project_dir: Path, step: str, message: str):
    """Log pipeline steps to file"""
    log_file = project_dir / "logs" / "pipeline.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] {step}: {message}\n")
    
    print(f"📝 {step}: {message}")

def validate_step_output(output_path: Path, expected_type: str = "files", min_count: int = 1) -> bool:
    """
    Validate that a pipeline step produced the expected output
    
    Args:
        output_path: Path to check for output
        expected_type: "files", "directories", or "any"
        min_count: Minimum number of items expected
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    if not output_path.exists():
        return False
    
    if expected_type == "files":
        items = list(output_path.glob("*.wav")) + list(output_path.glob("*.mp3"))
    elif expected_type == "directories":
        items = [p for p in output_path.iterdir() if p.is_dir()]
    else:  # "any"
        items = list(output_path.iterdir())
    
    return len(items) >= min_count

def check_speaker_directories(base_path: Path) -> dict:
    """Check if speaker directories exist and contain files"""
    results = {}
    for speaker in ["left_speaker", "right_speaker"]:
        speaker_path = base_path / speaker
        if speaker_path.exists():
            wav_files = list(speaker_path.glob("*.wav"))
            results[speaker] = len(wav_files)
        else:
            results[speaker] = 0
    return results


class VoicePipeline:
    """Voice Cloning Pipeline - Modular execution"""
    
    def __init__(self, project_dir: Optional[Union[str, Path]] = None):
        # Configuration
        self.INPUT_AUDIO = "/Users/adamleeperelman/Documents/LLM/Voice Clone/Raw_Sample/recording.mp3"
        self.EXTRACT_START = 0      # Start at 0 minutes
        self.EXTRACT_END = 15       # Extract first 15 minutes
        self.TEST_PHRASE = "Hello, this is a test of the fine-tuned voice model using multiple training samples for high quality synthesis."
        
        # State variables
        self.project_dir = None
        self.processor = None
        self.extracted_file = None
        self.channels = None
        self.separated_files = []
        self.filtered_files = []
        self.training_metadata = None
        self.reference_audio = None
        self.test_files = []
        self.generated_audio = None
        self.finetuned_model_dir = None
        self.finetuned_model_checkpoint = None
        self.f5_dataset_info = None
        
        if project_dir:
            self.load_existing_project(project_dir)

    def _collect_files(self, directory: Path, patterns: List[str]) -> List[Path]:
        if not directory.exists():
            return []
        files: List[Path] = []
        for pattern in patterns:
            files.extend(directory.glob(pattern))
        # Remove duplicates while preserving order by sorting unique set
        unique_files = sorted({file_path for file_path in files})
        return unique_files

    def _pick_latest_file(self, files: List[Path]) -> Optional[Path]:
        if not files:
            return None
        try:
            return max(files, key=lambda p: p.stat().st_mtime)
        except FileNotFoundError:
            return None

    def _looks_like_project(self, project_path: Path) -> bool:
        if not project_path.exists() or not project_path.is_dir():
            return False
        training_dir = project_path / "05_training_data"
        metadata_file = training_dir / "metadata.json"
        wavs_dir = training_dir / "wavs"
        if metadata_file.exists() or (wavs_dir.exists() and any(wavs_dir.glob("*.wav"))):
            return True

        legacy_dir = project_path / project_path.name / "05_training_data"
        legacy_metadata = legacy_dir / "metadata.json"
        legacy_wavs = legacy_dir / "wavs"
        return legacy_metadata.exists() or (legacy_wavs.exists() and any(legacy_wavs.glob("*.wav")))

    def _ensure_training_data_dir(self) -> Path:
        """Ensure the training data directory exists and consolidate legacy layouts."""

        if not self.project_dir:
            raise ValueError("Project directory is not set")

        training_dir = self.project_dir / "05_training_data"
        legacy_root = self.project_dir / self.project_dir.name
        legacy_dir = legacy_root / "05_training_data"

        if legacy_dir.exists():
            training_dir.mkdir(parents=True, exist_ok=True)
            for item in legacy_dir.iterdir():
                destination = training_dir / item.name
                if destination.exists():
                    continue
                shutil.move(str(item), str(destination))
            shutil.rmtree(legacy_dir)
            try:
                legacy_root.rmdir()
            except OSError:
                pass

        training_dir.mkdir(parents=True, exist_ok=True)
        return training_dir

    def _load_existing_state(self):
        # Reset relevant state before loading
        self.extracted_file = None
        self.channels = None
        self.separated_files = []
        self.filtered_files = []
        self.training_metadata = None
        self.reference_audio = None
        self.test_files = []
        self.generated_audio = None
        self.finetuned_model_dir = None
        self.finetuned_model_checkpoint = None

        if not self.project_dir:
            return

        # Extracted audio
        extracted_dir = self.project_dir / "01_extracted_audio"
        extracted_files = self._collect_files(extracted_dir, ["*.mp3", "*.wav"])
        latest_extracted = self._pick_latest_file(extracted_files)
        if latest_extracted:
            self.extracted_file = str(latest_extracted)

        # Stereo channels
        channels_dir = self.project_dir / "02_stereo_channels"
        left_channel = self._pick_latest_file(self._collect_files(channels_dir, ["*left*.wav", "*left*.mp3"]))
        right_channel = self._pick_latest_file(self._collect_files(channels_dir, ["*right*.wav", "*right*.mp3"]))
        if left_channel and right_channel:
            self.channels = {
                "left": str(left_channel),
                "right": str(right_channel)
            }

        # Voice separation outputs
        separated_dir = self.project_dir / "03_separated_voices" / "left_speaker"
        separated_files = self._collect_files(separated_dir, ["*.wav"])
        if separated_files:
            self.separated_files = [str(path) for path in separated_files]

        # Voice filtered outputs (fallback to separated if empty)
        filtered_dir = self.project_dir / "04_voice_filtered" / "left_speaker"
        filtered_files = self._collect_files(filtered_dir, ["*.wav"])
        if filtered_files:
            self.filtered_files = [str(path) for path in filtered_files]
        elif self.separated_files:
            self.filtered_files = list(self.separated_files)

        # Training metadata
        training_dir = self._ensure_training_data_dir()
        metadata_file = training_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    self.training_metadata = json.load(f)
            except json.JSONDecodeError:
                self.training_metadata = None

        if not self.training_metadata:
            wavs_dir = training_dir / "wavs"
            wav_files = self._collect_files(wavs_dir, ["*.wav"])
            if wav_files:
                self.training_metadata = {
                    "speaker": "main_speaker",
                    "total_samples": len(wav_files),
                    "total_duration": 0.0,
                    "average_duration": 0.0,
                    "samples": [
                        {
                            "audio_path": f"wavs/{path.name}",
                            "text": None,
                            "speaker": "main_speaker",
                            "duration": 0.0
                        }
                        for path in wav_files
                    ]
                }

        dataset_manifest = training_dir / "f5_dataset.json"
        if dataset_manifest.exists():
            try:
                with open(dataset_manifest, "r", encoding="utf-8") as f:
                    self.f5_dataset_info = json.load(f)
            except json.JSONDecodeError:
                self.f5_dataset_info = None

        # Existing fine-tuned models (if any)
        models_dir = self.project_dir / "06_models"
        checkpoint_files = self._collect_files(models_dir, ["*.pt", "*.safetensors"])
        latest_checkpoint = self._pick_latest_file(checkpoint_files)
        if latest_checkpoint:
            self.finetuned_model_dir = str(models_dir)
            self.finetuned_model_checkpoint = str(latest_checkpoint)

        # Generated outputs (optional)
        tests_dir = self.project_dir / "07_generated_audio" / "tests"
        self.test_files = [str(path) for path in self._collect_files(tests_dir, ["*.wav", "*.mp3"])]
        custom_phrase = self._pick_latest_file(self._collect_files(self.project_dir / "07_generated_audio", ["custom_phrase.*"]))
        if custom_phrase:
            self.generated_audio = str(custom_phrase)

    def load_existing_project(self, project_path: Union[str, Path]) -> Path:
        project_path = Path(project_path).expanduser().resolve()

        if not project_path.exists():
            raise ValueError(f"Existing project path not found: {project_path}")

        if not self._looks_like_project(project_path):
            raise ValueError(
                "The provided directory does not appear to contain prepared training data. "
                "Ensure the project has been processed at least through step 7."
            )

        self.project_dir = project_path
        self._load_existing_state()

        training_samples = self.training_metadata.get('total_samples', 0) if self.training_metadata else 0
        filtered_count = len(self.filtered_files)
        log_step(self.project_dir, "INIT", f"Resuming voice cloning pipeline in {self.project_dir}")
        log_step(
            self.project_dir,
            "RESUME",
            (
                f"Loaded existing project state - training samples: {training_samples}, "
                f"filtered clips: {filtered_count}, separated clips: {len(self.separated_files)}"
            )
        )

        return self.project_dir
    
    
    def step1_create_project(
        self,
        project_name: str = "voice_clone_project",
        existing_project_path: Optional[Union[str, Path]] = None
    ):
        """Step 1: Create project structure or resume from existing"""
        print("🎯 Step 1: Create Project Structure")

        if existing_project_path:
            return self.load_existing_project(existing_project_path)

        candidate_path = Path(project_name).expanduser()
        if candidate_path.exists() and self._looks_like_project(candidate_path):
            return self.load_existing_project(candidate_path)

        self.project_dir = create_project_structure(project_name)
        log_step(self.project_dir, "INIT", f"Starting voice cloning pipeline in {self.project_dir}")
        return self.project_dir
    
    
    def step2_initialize_processor(self):
        """Step 2: Initialize Voice AI Processor"""
        if not self.project_dir:
            raise ValueError("Must call step1_create_project() first")
        
        print("\n🎯 Step 2: Initialize Voice AI Processor")
        self.processor = create_processor(workspace_path=str(self.project_dir))
        log_step(self.project_dir, "PROCESSOR", "Voice AI Processor initialized")
        return self.processor
    
    
    def step3_extract_audio(self):
        """Step 3: Extract time range from input audio"""
        if not self.processor:
            raise ValueError("Must call step2_initialize_processor() first")
        
        print(f"\n🎯 Step 3: Extract Audio Range ({self.EXTRACT_START}-{self.EXTRACT_END} minutes)")
        
        if not os.path.exists(self.INPUT_AUDIO):
            print(f"❌ Input audio not found: {self.INPUT_AUDIO}")
            return None
        
        extracted_audio_path = self.project_dir / "01_extracted_audio" / f"recording_{self.EXTRACT_START}m_to_{self.EXTRACT_END}m.mp3"
        
        try:
            self.extracted_file = self.processor.extract_time_range(
                input_path=self.INPUT_AUDIO,
                start_minutes=self.EXTRACT_START,
                end_minutes=self.EXTRACT_END,
                output_path=str(extracted_audio_path)
            )
            
            # Validate extraction output
            if not os.path.exists(self.extracted_file):
                log_step(self.project_dir, "ERROR", f"Extraction failed - output file not created: {self.extracted_file}")
                return None
                
            # Check file size (should be reasonable for extracted audio)
            file_size_mb = os.path.getsize(self.extracted_file) / (1024 * 1024)
            if file_size_mb < 0.1:  # Less than 100KB seems wrong
                log_step(self.project_dir, "ERROR", f"Extracted file too small ({file_size_mb:.1f}MB) - possible extraction failure")
                return None
                
            log_step(self.project_dir, "EXTRACT", f"✅ Extracted {self.EXTRACT_END} minutes to {self.extracted_file} ({file_size_mb:.1f}MB)")
            return self.extracted_file
            
        except Exception as e:
            log_step(self.project_dir, "ERROR", f"Audio extraction failed: {e}")
            return None
    
    
    def step4_separate_channels(self):
        """Step 4: Separate stereo channels"""
        if not self.extracted_file:
            raise ValueError("Must call step3_extract_audio() first")
        
        print(f"\n🎯 Step 4: Separate Stereo Channels")
        
        channels_output = self.project_dir / "02_stereo_channels"
        
        try:
            self.channels = self.processor.separate_channels(
                input_path=self.extracted_file,
                output_dir=str(channels_output)
            )
            
            # Validate channel separation output
            channels_valid = True
            for channel_name in ['left', 'right']:  # Only check actual channel files
                channel_file = self.channels[channel_name]
                if not os.path.exists(channel_file):
                    log_step(self.project_dir, "ERROR", f"Channel separation failed - {channel_name} channel file not created: {channel_file}")
                    channels_valid = False
                else:
                    file_size_mb = os.path.getsize(channel_file) / (1024 * 1024)
                    if file_size_mb < 0.05:  # Very small file indicates issue
                        log_step(self.project_dir, "ERROR", f"{channel_name.title()} channel file too small ({file_size_mb:.1f}MB)")
                        channels_valid = False
                        
            if not channels_valid:
                return None
                
            log_step(self.project_dir, "CHANNELS", f"Channel separation complete")
            log_step(self.project_dir, "CHANNELS", f"Left channel: {self.channels['left']}")
            log_step(self.project_dir, "CHANNELS", f"Right channel: {self.channels['right']}")
            
            return self.channels
            
        except Exception as e:
            log_step(self.project_dir, "ERROR", f"Channel separation failed: {e}")
            return None
    
    
    def step5_extract_voice_segments(self):
        """Step 5: Process left channel to extract voice segments"""
        if not self.channels:
            raise ValueError("Must call step4_separate_channels() first")
        
        print(f"\n🎯 Step 5: Extract Voice Segments (Left Speaker Only)")
        
        separation_output = self.project_dir / "03_separated_voices"
        audio_for_separation = self.channels['left']
        
        # Define parameter sets from strict to very permissive
        parameter_sets = [
            {
                "name": "Standard",
                "min_duration": 8.0,
                "min_confidence": -0.5,
                "max_no_speech_prob": 0.3,
                "min_segment_len": 8.0,
                "silence_len": 2000
            },
            {
                "name": "Relaxed",
                "min_duration": 3.0,
                "min_confidence": -0.8,
                "max_no_speech_prob": 0.6,
                "min_segment_len": 3.0,
                "silence_len": 1500
            },
            {
                "name": "Permissive",
                "min_duration": 1.0,
                "min_confidence": -1.5,
                "max_no_speech_prob": 1.0,
                "min_segment_len": 1.0,
                "silence_len": 1000
            },
            {
                "name": "Very Permissive",
                "min_duration": 1.0,
                "min_confidence": -2.0,
                "max_no_speech_prob": 2.0,
                "min_segment_len": 1.0,
                "silence_len": 1000
            }
        ]
        
        # Try each parameter set until one succeeds
        for attempt, params in enumerate(parameter_sets, 1):
            print(f"\n   🔄 Attempt {attempt}: {params['name']} parameters")
            log_step(self.project_dir, "SEPARATION", f"Attempting voice segmentation with {params['name']} parameters")
            
            try:
                # Extract voice segments from left channel only
                separated_files = self.processor.extract_voice_segments(
                    input_path=audio_for_separation,
                    output_dir=str(separation_output / "left_speaker"),
                    min_duration=params["min_duration"],
                    silence_len=params["silence_len"],
                    silence_thresh=-35,
                    min_segment_len=params["min_segment_len"],
                    max_segment_len=20.0
                )
                
                # Validate output
                if separated_files and len(separated_files) > 0:
                    # Check if files actually exist
                    existing_files = [f for f in separated_files if os.path.exists(f)]
                    
                    if existing_files:
                        log_step(self.project_dir, "SEPARATION", f"✅ Success with {params['name']} parameters: {len(existing_files)} voice segments")
                        log_step(self.project_dir, "SEPARATION", f"Left speaker samples: {len(existing_files)}")
                        
                        self.separated_files = existing_files
                        return self.separated_files  # Success - exit retry loop
                    else:
                        log_step(self.project_dir, "SEPARATION", f"❌ Files reported but don't exist on disk")
                        separated_files = []
                else:
                    log_step(self.project_dir, "SEPARATION", f"❌ No segments generated with {params['name']} parameters")
                    
            except Exception as e:
                log_step(self.project_dir, "SEPARATION", f"❌ Error with {params['name']} parameters: {e}")
                separated_files = []
            
            # If this was the last attempt and still failed
            if not separated_files and attempt == len(parameter_sets):
                log_step(self.project_dir, "ERROR", "All voice separation attempts failed")
                return None
        
        # Validate we have output before continuing
        if not self.separated_files:
            log_step(self.project_dir, "ERROR", "Voice separation failed - no files generated after all attempts")
            return None
            
        return self.separated_files
    
    
    def step6_filter_voice_activity(self):
        """Step 6: Filter by voice activity (remove low voice or empty sounds)"""
        if not self.separated_files:
            raise ValueError("Must call step5_extract_voice_segments() first")
        
        print(f"\n🎯 Step 6: Filter by Voice Activity")
        
        voice_filtered_output = self.project_dir / "04_voice_filtered"
        separation_output = self.project_dir / "03_separated_voices"
        left_speaker_source = separation_output / "left_speaker"
        
        # Validate input exists
        if not left_speaker_source.exists():
            log_step(self.project_dir, "ERROR", f"Left speaker directory not found: {left_speaker_source}")
            return None
        
        audio_files = list(left_speaker_source.glob("*.wav"))
        if not audio_files:
            log_step(self.project_dir, "ERROR", f"No WAV files found in {left_speaker_source}")
            return None
        
        log_step(self.project_dir, "VOICE_FILTER", f"Found {len(audio_files)} files for voice activity filtering")
        
        # Voice activity filtering parameter sets (from strict to permissive)
        filter_parameter_sets = [
            {
                "name": "Standard",
                "min_voice_threshold": 0.02,
                "min_voice_ratio": 0.3
            },
            {
                "name": "Relaxed",
                "min_voice_threshold": 0.015,
                "min_voice_ratio": 0.25
            },
            {
                "name": "Permissive",
                "min_voice_threshold": 0.01,
                "min_voice_ratio": 0.2
            },
            {
                "name": "Very Permissive",
                "min_voice_threshold": 0.005,
                "min_voice_ratio": 0.1
            }
        ]
        
        # Try each filter parameter set
        for attempt, params in enumerate(filter_parameter_sets, 1):
            print(f"\n   🔄 Voice Filter Attempt {attempt}: {params['name']} parameters")
            log_step(self.project_dir, "VOICE_FILTER", f"Attempting voice filtering with {params['name']} parameters")
            
            try:
                filtered_files = self.processor.filter_audio_by_voice_activity(
                    input_paths=[str(f) for f in audio_files],
                    output_dir=str(voice_filtered_output / "left_speaker"),
                    min_voice_threshold=params["min_voice_threshold"],
                    min_voice_ratio=params["min_voice_ratio"]
                )
                
                # Validate output
                if filtered_files and len(filtered_files) > 0:
                    # Check if files actually exist
                    existing_filtered = [f for f in filtered_files if os.path.exists(f)]
                    
                    if existing_filtered:
                        log_step(self.project_dir, "VOICE_FILTER", f"✅ Success with {params['name']} parameters: {len(existing_filtered)} files kept from {len(audio_files)}")
                        log_step(self.project_dir, "VOICE_FILTER", f"Filtered out {len(audio_files) - len(existing_filtered)} files with low/no voice activity")
                        
                        # Update source directory for training data preparation
                        self.filtered_files = existing_filtered
                        return self.filtered_files  # Success - exit retry loop
                    else:
                        log_step(self.project_dir, "VOICE_FILTER", f"❌ Files reported but don't exist on disk")
                        filtered_files = []
                else:
                    log_step(self.project_dir, "VOICE_FILTER", f"❌ No files passed {params['name']} filtering")
                    
            except Exception as e:
                log_step(self.project_dir, "VOICE_FILTER", f"❌ Error with {params['name']} parameters: {e}")
                filtered_files = []
        
        # If all filtering attempts failed, continue with original files
        if not self.filtered_files:
            log_step(self.project_dir, "WARNING", "All voice filtering attempts removed all files - continuing with original separated files")
            self.filtered_files = self.separated_files
        
        return self.filtered_files
    
    
    def step7_prepare_training_data(self):
        """Step 7: Prepare training data from filtered audio"""
        if not self.filtered_files and not self.separated_files:
            raise ValueError("Must call step6_filter_voice_activity() or step5_extract_voice_segments() first")
        
        print(f"\n🎯 Step 7: Prepare Training Data")

        training_output = self._ensure_training_data_dir()

        # Determine source directory
        if self.filtered_files:
            left_speaker_source = self.project_dir / "04_voice_filtered" / "left_speaker"
        else:
            left_speaker_source = self.project_dir / "03_separated_voices" / "left_speaker"

        # Validate input directory and files
        if not left_speaker_source.exists():
            log_step(self.project_dir, "ERROR", f"Left speaker directory not found: {left_speaker_source}")
            return None

        training_files = list(left_speaker_source.glob("*.wav"))
        if not training_files:
            log_step(self.project_dir, "ERROR", f"No WAV files found for training in {left_speaker_source}")
            return None

        log_step(self.project_dir, "TRAINING", f"Found {len(training_files)} files for training data preparation")

        try:
            self.training_metadata = self.processor.prepare_training_data(
                source_dir=str(left_speaker_source),
                output_dir="05_training_data",
                speaker_name="main_speaker"
            )

            if self.training_metadata:
                log_step(self.project_dir, "TRAINING_PREP", f"Training data prepared: {self.training_metadata['total_samples']} samples")
                log_step(self.project_dir, "TRAINING_PREP", f"Total duration: {self.training_metadata['total_duration']:.1f} seconds")

                # Validate training data
                validation = self.processor.validate_training_data("05_training_data")
                if validation.get("valid", True):
                    log_step(self.project_dir, "VALIDATION", f"Training data validation passed: {validation['total_samples']} samples")
                    if validation.get("issues"):
                        log_step(self.project_dir, "VALIDATION", f"Minor issues found: {len(validation['issues'])}")
                else:
                    log_step(self.project_dir, "WARNING", f"Training data validation issues: {validation.get('errors', 'Unknown')}")

                return self.training_metadata
            else:
                log_step(self.project_dir, "ERROR", "Training data preparation failed")
                return None

        except Exception as e:
            log_step(self.project_dir, "ERROR", f"Training data preparation failed: {e}")
            return None
    
    
    def step8_fine_tune_model(self, use_base_model=False, quick_test=True):
        """Step 8: Fine-tune voice model"""
        if not self.training_metadata:
            raise ValueError("Must call step7_prepare_training_data() first")
        
        print(f"\n🎯 Step 8: Fine-tune Voice Model")
        
        training_output = self._ensure_training_data_dir()
        models_output = self.project_dir / "06_models"
        models_output.mkdir(exist_ok=True)
        
        # Check what training data we have
        training_files = list((training_output / "wavs").glob("*.wav"))
        log_step(self.project_dir, "FINETUNE", f"Training data available: {len(training_files)} audio files")

        def _base_model_response(note: str, success: bool = True, error: Optional[str] = None):
            log_step(self.project_dir, "FINETUNE", "⚠️  Using F5-TTS Base model (no fine-tuning)")
            log_step(self.project_dir, "FINETUNE", f"💡 {note}")
            self.finetuned_model_dir = None
            self.finetuned_model_checkpoint = None
            response = {
                "success": success,
                "message": "Using base F5-TTS model (no fine-tuning performed)" if success else "Training failed - using base model",
                "model_used": "F5TTS_Base",
                "training_data_prepared": len(training_files)
            }
            if success:
                response["note"] = note
            if error:
                response["error"] = error
            return response

        if use_base_model:
            return _base_model_response("Fine-tuning skipped per configuration")

        # Export dataset in the format expected by F5-TTS
        dataset_manifest = training_output / "f5_dataset.json"
        dataset_seed = self.project_dir.name if self.project_dir else "voice_clone_project"

        try:
            dataset_info = self.processor.prepare_f5_dataset(
                training_dir="05_training_data",
                dataset_name=dataset_seed,
                tokenizer="byte",
                refresh=True
            )
            self.f5_dataset_info = dataset_info
            with open(dataset_manifest, "w", encoding="utf-8") as f:
                json.dump(dataset_info, f, indent=2)

            log_step(
                self.project_dir,
                "FINETUNE",
                (
                    f"Dataset exported for F5-TTS: {dataset_info['dataset_name']} "
                    f"(samples={dataset_info['num_samples']}, tokenizer={dataset_info['tokenizer']})"
                )
            )
        except Exception as e:
            log_step(self.project_dir, "FINETUNE", f"❌ Dataset export failed: {e}")
            return _base_model_response("Dataset export failed; using base model", success=False, error=str(e))

        # Attempt actual F5-TTS training
        log_step(self.project_dir, "FINETUNE", "🔬 Attempting F5-TTS fine-tuning...")

        try:
            from f5_tts.train.train import main as train_main
            from omegaconf import OmegaConf

            base_config_path = files("f5_tts").joinpath("configs/F5TTS_Base.yaml")
            config = OmegaConf.load(str(base_config_path))

            config.model.tokenizer = dataset_info["tokenizer"]
            config.datasets.name = dataset_info["dataset_name"]
            config.datasets.batch_size_type = "sample"
            config.datasets.batch_size_per_gpu = 2 if quick_test else 6
            config.datasets.num_workers = 2 if quick_test else 8
            config.datasets.max_samples = 0

            config.optim.epochs = 1 if quick_test else 10
            config.optim.learning_rate = 2e-5 if quick_test else 7.5e-5
            config.optim.grad_accumulation_steps = 1
            config.optim.num_warmup_updates = max(10, len(training_files) * 2)
            config.optim.bnb_optimizer = False

            config.ckpts.logger = None
            config.ckpts.log_samples = False
            config.ckpts.save_per_updates = max(10, len(training_files))
            config.ckpts.last_per_updates = max(10, len(training_files))
            config.ckpts.keep_last_n_checkpoints = 2
            config.ckpts.save_dir = f"ckpts/{self.project_dir.name}"
            hydra_run_dir = (models_output / f"hydra_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()
            config.hydra.run.dir = str(hydra_run_dir)

            log_step(
                self.project_dir,
                "FINETUNE",
                f"Starting training: epochs={config.optim.epochs}, lr={config.optim.learning_rate}, batch={config.datasets.batch_size_per_gpu}"
            )

            train_main(config)

            log_step(self.project_dir, "FINETUNE", "✅ F5-TTS training completed!")

            ckpt_root = Path(files("f5_tts").joinpath(f"../../{config.ckpts.save_dir}"))
            project_ckpt_dir = models_output / "f5_tts"
            project_ckpt_dir.mkdir(parents=True, exist_ok=True)

            copied_files: List[Path] = []
            if ckpt_root.exists():
                for ckpt in ckpt_root.glob("*.pt"):
                    destination = project_ckpt_dir / ckpt.name
                    shutil.copy2(ckpt, destination)
                    copied_files.append(destination)
                for ckpt in ckpt_root.glob("*.safetensors"):
                    destination = project_ckpt_dir / ckpt.name
                    shutil.copy2(ckpt, destination)
                    copied_files.append(destination)

            best_checkpoint_path = None
            if copied_files:
                best_checkpoint_path = max(copied_files, key=lambda p: p.stat().st_mtime)
                log_step(self.project_dir, "FINETUNE", f"Checkpoint available: {best_checkpoint_path}")
            else:
                log_step(self.project_dir, "FINETUNE", "⚠️  No checkpoint files copied to project directory")

            self.finetuned_model_dir = str(project_ckpt_dir)
            self.finetuned_model_checkpoint = str(best_checkpoint_path) if best_checkpoint_path else None

            return {
                "success": True,
                "message": "F5-TTS training completed",
                "model_used": "F5TTS_Custom",
                "epochs": config.optim.epochs,
                "dataset": dataset_info,
                "checkpoint_dir": self.finetuned_model_dir,
                "checkpoint_file": self.finetuned_model_checkpoint
            }

        except ImportError as e:
            log_step(self.project_dir, "FINETUNE", f"❌ Training imports failed: {e}")
            return _base_model_response("F5-TTS training modules unavailable", success=False, error=str(e))
        except Exception as e:
            log_step(self.project_dir, "FINETUNE", f"❌ Training failed: {e}")
            return _base_model_response("Fine-tuning encountered an error", success=False, error=str(e))
    
    
    def step9_test_voice_cloning(self):
        """Step 9: Test voice cloning with existing samples"""
        if not self.processor:
            raise ValueError("Must call step2_initialize_processor() first")
        
        print(f"\n🎯 Step 9: Test Voice Cloning")
        
        test_output = self.project_dir / "07_generated_audio" / "tests"
        test_output.mkdir(parents=True, exist_ok=True)
        
        # Determine source directory for reference audio
        if self.filtered_files:
            left_speaker_source = self.project_dir / "04_voice_filtered" / "left_speaker"
        elif self.separated_files:
            left_speaker_source = self.project_dir / "03_separated_voices" / "left_speaker"
        else:
            log_step(self.project_dir, "WARNING", "No voice segments available for testing")
            return None
        
        try:
            # Find the best quality reference audio from left speaker
            self.reference_audio = self.processor.find_reference_audio(str(left_speaker_source))
            
            if self.reference_audio:
                log_step(self.project_dir, "REFERENCE", f"Using reference audio: {Path(self.reference_audio).name}")
                
                # Run voice cloning tests
                self.test_files = self.processor.test_voice_cloning(
                    reference_audio=self.reference_audio,
                    output_dir=str(test_output)
                )
                
                if self.test_files:
                    log_step(self.project_dir, "TEST", f"Voice cloning tests complete: {len(self.test_files)} files generated")
                    return self.test_files
                else:
                    log_step(self.project_dir, "WARNING", "Voice cloning tests failed")
                    return None
            else:
                log_step(self.project_dir, "WARNING", "No suitable reference audio found")
                return None
                
        except Exception as e:
            log_step(self.project_dir, "ERROR", f"Voice cloning tests failed: {e}")
            return None
    
    
    def step10_synthesize_custom_phrase(self):
        """Step 10: Synthesize custom phrase"""
        if not self.reference_audio:
            # Try to find reference audio if not already set
            self.step9_test_voice_cloning()
        
        if not self.reference_audio:
            log_step(self.project_dir, "WARNING", "Cannot synthesize - no reference audio available")
            return None
        
        print(f"\n🎯 Step 10: Synthesize Custom Phrase")
        
        synthesis_output = self.project_dir / "07_generated_audio"
        
        try:
            # Generate the custom phrase
            custom_audio_path = synthesis_output / "custom_phrase.wav"
            
            self.generated_audio = self.processor.synthesize_speech(
                text=self.TEST_PHRASE,
                reference_audio=self.reference_audio,
                model="F5-TTS",  # Using base model since fine-tuning setup is prepared but not executed
                output_path=str(custom_audio_path)
            )
            
            if self.generated_audio and os.path.exists(self.generated_audio):
                file_size = os.path.getsize(self.generated_audio) / 1024  # KB
                log_step(self.project_dir, "SYNTHESIS", f"Custom phrase synthesized: {Path(self.generated_audio).name}")
                log_step(self.project_dir, "SYNTHESIS", f"File size: {file_size:.1f} KB")
                log_step(self.project_dir, "SYNTHESIS", f"Text: '{self.TEST_PHRASE[:50]}...'")
                return self.generated_audio
            else:
                log_step(self.project_dir, "WARNING", "Custom phrase synthesis failed")
                return None
                
        except Exception as e:
            log_step(self.project_dir, "ERROR", f"Custom phrase synthesis failed: {e}")
            return None
    
    
    def step11_generate_report(self):
        """Step 11: Generate final project report"""
        if not self.project_dir:
            raise ValueError("Must call step1_create_project() first")
        
        print(f"\n🎯 Step 11: Generate Project Report")
        
        try:
            # Get final workspace status
            status = self.processor.get_workspace_status() if self.processor else {}
            
            # Create comprehensive report
            report = {
                "project_info": {
                    "name": self.project_dir.name,
                    "created": datetime.now().isoformat(),
                    "input_audio": self.INPUT_AUDIO,
                    "time_range": f"{self.EXTRACT_START}-{self.EXTRACT_END} minutes"
                },
                "pipeline_results": {
                    "audio_extraction": {
                        "success": bool(self.extracted_file),
                        "output_file": str(self.extracted_file) if self.extracted_file else None
                    },
                    "voice_separation": {
                        "success": len(self.separated_files) > 0,
                        "files_generated": len(self.separated_files),
                        "left_speaker_samples": status.get('separation', {}).get('left_speaker_files', 0),
                        "right_speaker_samples": status.get('separation', {}).get('right_speaker_files', 0)
                    },
                    "training_preparation": {
                        "success": bool(self.training_metadata),
                        "total_samples": self.training_metadata.get('total_samples', 0) if self.training_metadata else 0,
                        "total_duration": self.training_metadata.get('total_duration', 0) if self.training_metadata else 0
                    },
                    "voice_synthesis": {
                        "test_files_generated": len(self.test_files) if self.test_files else 0,
                        "custom_phrase_generated": bool(self.generated_audio)
                    }
                },
                "workspace_status": status,
                "test_phrase": self.TEST_PHRASE
            }
            
            # Save report
            report_file = self.project_dir / "PROJECT_REPORT.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            log_step(self.project_dir, "REPORT", f"Project report saved: {report_file}")
            return report
            
        except Exception as e:
            log_step(self.project_dir, "ERROR", f"Report generation failed: {e}")
            return None
    
    
    def run_complete_pipeline(self):
        """Run the complete pipeline from start to finish"""
        print("🚀 Voice Cloning Pipeline - Complete Workflow")
        print("=" * 60)
        
        # Run all steps
        self.step1_create_project()
        self.step2_initialize_processor()
        self.step3_extract_audio()
        self.step4_separate_channels()
        self.step5_extract_voice_segments()
        self.step6_filter_voice_activity()
        self.step7_prepare_training_data()
        self.step8_fine_tune_model()
        self.step9_test_voice_cloning()
        self.step10_synthesize_custom_phrase()
        report = self.step11_generate_report()
        
        # Final summary
        print(f"\n🎉 Pipeline Complete!")
        print("=" * 60)
        print(f"📁 Project Directory: {self.project_dir}")
        print(f"📄 Report: {self.project_dir}/PROJECT_REPORT.json")
        print(f"📋 Logs: {self.project_dir}/logs/pipeline.log")
        
        # List key output files
        print(f"\n📂 Key Output Files:")
        if self.extracted_file:
            print(f"  🎵 Extracted Audio: {Path(self.extracted_file).name}")
        
        if self.separated_files:
            print(f"  🎤 Separated Voices: {len(self.separated_files)} files in 03_separated_voices/")
        
        if self.training_metadata:
            print(f"  🎯 Training Data: {self.training_metadata['total_samples']} samples in 05_training_data/")
        
        if self.test_files:
            print(f"  🧪 Test Audio: {len(self.test_files)} files in 07_generated_audio/tests/")
        
        if self.generated_audio and os.path.exists(self.generated_audio):
            print(f"  🗣️ Custom Phrase: {Path(self.generated_audio).name}")
        
        print(f"\n💡 Next Steps:")
        print(f"  1. Review generated samples in the project directory")
        print(f"  2. Check the pipeline log for detailed information")
        print(f"  3. For actual fine-tuning, refer to F5-TTS documentation")
        print(f"  4. Experiment with different time ranges or parameters")
        
        return report

# %%
# Example: Run the complete pipeline
# pipeline = VoicePipeline()
# pipeline.run_complete_pipeline()

if __name__ == "__main__":
    # Example: Run individual steps
    pipeline = VoicePipeline()

    # Step 1: Create project structure
    pipeline.step1_create_project()

    # Step 2: Initialize processor
    pipeline.step2_initialize_processor()

    # Step 3: Extract audio range
    pipeline.step3_extract_audio()

    # Step 4: Separate stereo channels
    pipeline.step4_separate_channels()

    # Step 5: Extract voice segments
    pipeline.step5_extract_voice_segments()

    # Step 6: Filter by voice activity
    pipeline.step6_filter_voice_activity()

    # Step 7: Prepare training data
    pipeline.step7_prepare_training_data()

    # Step 8: Fine-tune model
    pipeline.step8_fine_tune_model()

    # Step 9: Test voice cloning
    pipeline.step9_test_voice_cloning()

    # Step 10: Synthesize custom phrase
    pipeline.step10_synthesize_custom_phrase()

    # Step 11: Generate report
    pipeline.step11_generate_report()
