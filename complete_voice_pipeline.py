#!/usr/bin/env python3
"""
Complete Voice Cloning Pipeline
End-to-end script using the modular Voice AI system
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Import the voice AI modules
from voice_ai import create_processor

def create_project_structure(project_name: str = "voice_clone_project") -> Path:
    """Create a new project structure for the voice cloning pipeline"""
    
    # Create project directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_dir = Path(f"{project_name}_{timestamp}")
    
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

def main():
    """Main pipeline execution"""
    
    print("🚀 Voice Cloning Pipeline - Complete Workflow")
    print("=" * 60)
    
    # Configuration
    INPUT_AUDIO = "/Users/adamleeperelman/Documents/LLM/Voice Clone/Raw_Sample/recording.mp3"
    EXTRACT_START = 0      # Start at 0 minutes
    EXTRACT_END = 15       # Extract first 15 minutes
    TEST_PHRASE = "Hello, this is a test of the fine-tuned voice model using multiple training samples for high quality synthesis."
    
    # Step 1: Create project structure
    project_dir = create_project_structure()
    log_step(project_dir, "INIT", f"Starting voice cloning pipeline in {project_dir}")
    
    # Step 2: Initialize processor
    print("\n🎯 Step 1: Initialize Voice AI Processor")
    processor = create_processor(workspace_path=str(project_dir))
    log_step(project_dir, "PROCESSOR", "Voice AI Processor initialized")
    
    # Step 3: Extract time range
    print(f"\n🎯 Step 2: Extract Audio Range ({EXTRACT_START}-{EXTRACT_END} minutes)")
    
    if not os.path.exists(INPUT_AUDIO):
        print(f"❌ Input audio not found: {INPUT_AUDIO}")
        return
    
    extracted_audio_path = project_dir / "01_extracted_audio" / f"recording_{EXTRACT_START}m_to_{EXTRACT_END}m.mp3"
    
    try:
        extracted_file = processor.extract_time_range(
            input_path=INPUT_AUDIO,
            start_minutes=EXTRACT_START,
            end_minutes=EXTRACT_END,
            output_path=str(extracted_audio_path)
        )
        
        # Validate extraction output
        if not os.path.exists(extracted_file):
            log_step(project_dir, "ERROR", f"Extraction failed - output file not created: {extracted_file}")
            return
            
        # Check file size (should be reasonable for extracted audio)
        file_size_mb = os.path.getsize(extracted_file) / (1024 * 1024)
        if file_size_mb < 0.1:  # Less than 100KB seems wrong
            log_step(project_dir, "ERROR", f"Extracted file too small ({file_size_mb:.1f}MB) - possible extraction failure")
            return
            
        log_step(project_dir, "EXTRACT", f"✅ Extracted {EXTRACT_END} minutes to {extracted_file} ({file_size_mb:.1f}MB)")
        
    except Exception as e:
        log_step(project_dir, "ERROR", f"Audio extraction failed: {e}")
        return
    except Exception as e:
        log_step(project_dir, "ERROR", f"Time extraction failed: {e}")
        return
    
    # Step 4: Separate stereo channels
    print(f"\n🎯 Step 3: Separate Stereo Channels")
    
    channels_output = project_dir / "02_stereo_channels"
    
    try:
        channels = processor.separate_channels(
            input_path=extracted_file,
            output_dir=str(channels_output)
        )
        
        # Validate channel separation output
        channels_valid = True
        for channel_name in ['left', 'right']:  # Only check actual channel files
            channel_file = channels[channel_name]
            if not os.path.exists(channel_file):
                log_step(project_dir, "ERROR", f"Channel separation failed - {channel_name} channel file not created: {channel_file}")
                channels_valid = False
            else:
                file_size_mb = os.path.getsize(channel_file) / (1024 * 1024)
                if file_size_mb < 0.05:  # Very small file indicates issue
                    log_step(project_dir, "ERROR", f"{channel_name.title()} channel file too small ({file_size_mb:.1f}MB)")
                    channels_valid = False
                    
        if not channels_valid:
            return
            
        log_step(project_dir, "CHANNELS", f"Channel separation complete")
        log_step(project_dir, "CHANNELS", f"Left channel: {channels['left']}")
        log_step(project_dir, "CHANNELS", f"Right channel: {channels['right']}")
        
        # Use left channel only (contains the target speaker)
        audio_for_separation = channels['left']
        
    except Exception as e:
        log_step(project_dir, "ERROR", f"Channel separation failed: {e}")
        return
    
    # Step 4: Process left channel to extract voice segments
    print(f"\n🎯 Step 4: Extract Voice Segments (Left Speaker Only)")
    
    separation_output = project_dir / "03_separated_voices"
    separated_files = []
    
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
        log_step(project_dir, "SEPARATION", f"Attempting voice segmentation with {params['name']} parameters")
        
        try:
            # Extract voice segments from left channel only
            separated_files = processor.extract_voice_segments(
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
                    log_step(project_dir, "SEPARATION", f"✅ Success with {params['name']} parameters: {len(existing_files)} voice segments")
                    log_step(project_dir, "SEPARATION", f"Left speaker samples: {len(existing_files)}")
                    
                    separated_files = existing_files
                    break  # Success - exit retry loop
                else:
                    log_step(project_dir, "SEPARATION", f"❌ Files reported but don't exist on disk")
                    separated_files = []
            else:
                log_step(project_dir, "SEPARATION", f"❌ No segments generated with {params['name']} parameters")
                
        except Exception as e:
            log_step(project_dir, "SEPARATION", f"❌ Error with {params['name']} parameters: {e}")
            separated_files = []
        
        # If this was the last attempt and still failed
        if not separated_files and attempt == len(parameter_sets):
            log_step(project_dir, "ERROR", "All voice separation attempts failed")
            return
    
    # Validate we have output before continuing
    if not separated_files:
        log_step(project_dir, "ERROR", "Voice separation failed - no files generated after all attempts")
        return
    
    # Step 5: Filter by voice activity (remove low voice or empty sounds)
    print(f"\n🎯 Step 5: Filter by Voice Activity")
    
    voice_filtered_output = project_dir / "04_voice_filtered"
    left_speaker_source = separation_output / "left_speaker"
    
    # Validate input exists
    if not left_speaker_source.exists():
        log_step(project_dir, "ERROR", f"Left speaker directory not found: {left_speaker_source}")
        return
    
    audio_files = list(left_speaker_source.glob("*.wav"))
    if not audio_files:
        log_step(project_dir, "ERROR", f"No WAV files found in {left_speaker_source}")
        return
    
    log_step(project_dir, "VOICE_FILTER", f"Found {len(audio_files)} files for voice activity filtering")
    
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
    
    filtered_files = []
    
    # Try each filter parameter set
    for attempt, params in enumerate(filter_parameter_sets, 1):
        print(f"\n   🔄 Voice Filter Attempt {attempt}: {params['name']} parameters")
        log_step(project_dir, "VOICE_FILTER", f"Attempting voice filtering with {params['name']} parameters")
        
        try:
            filtered_files = processor.filter_audio_by_voice_activity(
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
                    log_step(project_dir, "VOICE_FILTER", f"✅ Success with {params['name']} parameters: {len(existing_filtered)} files kept from {len(audio_files)}")
                    log_step(project_dir, "VOICE_FILTER", f"Filtered out {len(audio_files) - len(existing_filtered)} files with low/no voice activity")
                    
                    # Update source directory for training data preparation
                    left_speaker_source = voice_filtered_output / "left_speaker"
                    filtered_files = existing_filtered
                    break  # Success - exit retry loop
                else:
                    log_step(project_dir, "VOICE_FILTER", f"❌ Files reported but don't exist on disk")
                    filtered_files = []
            else:
                log_step(project_dir, "VOICE_FILTER", f"❌ No files passed {params['name']} filtering")
                
        except Exception as e:
            log_step(project_dir, "VOICE_FILTER", f"❌ Error with {params['name']} parameters: {e}")
            filtered_files = []
    
    # If all filtering attempts failed, continue with original files
    if not filtered_files:
        log_step(project_dir, "WARNING", "All voice filtering attempts removed all files - continuing with original separated files")
        left_speaker_source = separation_output / "left_speaker"
    
    # Step 6: Prepare training data from filtered left speaker
    print(f"\n🎯 Step 6: Prepare Training Data")
    
    training_output = project_dir / "05_training_data"
    
    # Validate input directory and files
    if not left_speaker_source.exists():
        log_step(project_dir, "ERROR", f"Left speaker directory not found: {left_speaker_source}")
        return
    
    training_files = list(left_speaker_source.glob("*.wav"))
    if not training_files:
        log_step(project_dir, "ERROR", f"No WAV files found for training in {left_speaker_source}")
        return
    
    log_step(project_dir, "TRAINING", f"Found {len(training_files)} files for training data preparation")
    
    try:
        training_metadata = processor.prepare_training_data(
            source_dir=str(left_speaker_source),
            output_dir=str(training_output),
            speaker_name="main_speaker"
        )
        
        if training_metadata:
            log_step(project_dir, "TRAINING_PREP", f"Training data prepared: {training_metadata['total_samples']} samples")
            log_step(project_dir, "TRAINING_PREP", f"Total duration: {training_metadata['total_duration']:.1f} seconds")
            
            # Validate training data
            validation = processor.validate_training_data(str(training_output))
            if validation.get("valid", True):
                log_step(project_dir, "VALIDATION", f"Training data validation passed: {validation['total_samples']} samples")
                if validation.get("issues"):
                    log_step(project_dir, "VALIDATION", f"Minor issues found: {len(validation['issues'])}")
            else:
                log_step(project_dir, "WARNING", f"Training data validation issues: {validation.get('errors', 'Unknown')}")
        else:
            log_step(project_dir, "ERROR", "Training data preparation failed")
            return
            
    except Exception as e:
        log_step(project_dir, "ERROR", f"Training data preparation failed: {e}")
        return
    
    # Step 7: Fine-tune model
    print(f"\n🎯 Step 7: Fine-tune Voice Model")
    
    try:
        # Note: This prepares the fine-tuning setup since actual F5-TTS fine-tuning 
        # requires specific implementation that may vary by installation
        finetune_result = processor.fine_tune_model(
            training_dir=str(training_output),
            model_name="main_speaker_model",
            epochs=100,
            learning_rate=1e-4,
            batch_size=4
        )
        
        if finetune_result.get("success", False):
            log_step(project_dir, "FINETUNE", f"Fine-tuning setup complete: {finetune_result['model_name']}")
            log_step(project_dir, "FINETUNE", f"Script saved: {finetune_result.get('script_path', 'N/A')}")
        else:
            log_step(project_dir, "FINETUNE", "Fine-tuning setup prepared (manual execution may be required)")
            
    except Exception as e:
        log_step(project_dir, "ERROR", f"Fine-tuning setup failed: {e}")
        # Continue with synthesis using original model
    
    # Step 8: Test voice cloning with existing samples
    print(f"\n🎯 Step 8: Test Voice Cloning")
    
    test_output = project_dir / "07_generated_audio" / "tests"
    test_output.mkdir(parents=True, exist_ok=True)
    
    try:
        # Find the best quality reference audio from left speaker
        reference_audio = processor.find_reference_audio(str(left_speaker_source))
        
        if reference_audio:
            log_step(project_dir, "REFERENCE", f"Using reference audio: {Path(reference_audio).name}")
            
            # Run voice cloning tests
            test_files = processor.test_voice_cloning(
                reference_audio=reference_audio,
                output_dir=str(test_output)
            )
            
            if test_files:
                log_step(project_dir, "TEST", f"Voice cloning tests complete: {len(test_files)} files generated")
            else:
                log_step(project_dir, "WARNING", "Voice cloning tests failed")
        else:
            log_step(project_dir, "WARNING", "No suitable reference audio found")
            
    except Exception as e:
        log_step(project_dir, "ERROR", f"Voice cloning tests failed: {e}")
    
    # Step 9: Synthesize custom phrase
    print(f"\n🎯 Step 9: Synthesize Custom Phrase")
    
    synthesis_output = project_dir / "07_generated_audio"
    
    try:
        if reference_audio:
            # Generate the custom phrase
            custom_audio_path = synthesis_output / "custom_phrase.wav"
            
            generated_audio = processor.synthesize_speech(
                text=TEST_PHRASE,
                reference_audio=reference_audio,
                model="F5-TTS",  # Using base model since fine-tuning setup is prepared but not executed
                output_path=str(custom_audio_path)
            )
            
            if generated_audio and os.path.exists(generated_audio):
                file_size = os.path.getsize(generated_audio) / 1024  # KB
                log_step(project_dir, "SYNTHESIS", f"Custom phrase synthesized: {Path(generated_audio).name}")
                log_step(project_dir, "SYNTHESIS", f"File size: {file_size:.1f} KB")
                log_step(project_dir, "SYNTHESIS", f"Text: '{TEST_PHRASE[:50]}...'")
            else:
                log_step(project_dir, "WARNING", "Custom phrase synthesis failed")
        else:
            log_step(project_dir, "WARNING", "Cannot synthesize - no reference audio available")
            
    except Exception as e:
        log_step(project_dir, "ERROR", f"Custom phrase synthesis failed: {e}")
    
    # Step 10: Generate final report
    print(f"\n🎯 Step 8: Generate Project Report")
    
    try:
        # Get final workspace status
        status = processor.get_workspace_status()
        
        # Create comprehensive report
        report = {
            "project_info": {
                "name": project_dir.name,
                "created": datetime.now().isoformat(),
                "input_audio": INPUT_AUDIO,
                "time_range": f"{EXTRACT_START}-{EXTRACT_END} minutes"
            },
            "pipeline_results": {
                "audio_extraction": {
                    "success": os.path.exists(extracted_file) if 'extracted_file' in locals() else False,
                    "output_file": str(extracted_audio_path) if 'extracted_file' in locals() else None
                },
                "voice_separation": {
                    "success": len(separated_files) > 0 if 'separated_files' in locals() else False,
                    "files_generated": len(separated_files) if 'separated_files' in locals() else 0,
                    "left_speaker_samples": status['separation'].get('left_speaker_files', 0),
                    "right_speaker_samples": status['separation'].get('right_speaker_files', 0)
                },
                "training_preparation": {
                    "success": bool(training_metadata) if 'training_metadata' in locals() else False,
                    "total_samples": training_metadata.get('total_samples', 0) if 'training_metadata' in locals() else 0,
                    "total_duration": training_metadata.get('total_duration', 0) if 'training_metadata' in locals() else 0
                },
                "voice_synthesis": {
                    "test_files_generated": len(test_files) if 'test_files' in locals() else 0,
                    "custom_phrase_generated": os.path.exists(custom_audio_path) if 'custom_audio_path' in locals() else False
                }
            },
            "workspace_status": status,
            "test_phrase": TEST_PHRASE
        }
        
        # Save report
        report_file = project_dir / "PROJECT_REPORT.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        log_step(project_dir, "REPORT", f"Project report saved: {report_file}")
        
    except Exception as e:
        log_step(project_dir, "ERROR", f"Report generation failed: {e}")
    
    # Final summary
    print(f"\n🎉 Pipeline Complete!")
    print("=" * 60)
    print(f"📁 Project Directory: {project_dir}")
    print(f"📄 Report: {project_dir}/PROJECT_REPORT.json")
    print(f"📋 Logs: {project_dir}/logs/pipeline.log")
    
    # List key output files
    print(f"\n📂 Key Output Files:")
    if 'extracted_file' in locals():
        print(f"  🎵 Extracted Audio: {Path(extracted_file).name}")
    
    if 'separated_files' in locals() and separated_files:
        print(f"  🎤 Separated Voices: {len(separated_files)} files in 02_separated_voices/")
    
    if 'training_metadata' in locals() and training_metadata:
        print(f"  🎯 Training Data: {training_metadata['total_samples']} samples in 03_training_data/")
    
    if 'test_files' in locals() and test_files:
        print(f"  🧪 Test Audio: {len(test_files)} files in 05_generated_audio/tests/")
    
    if 'generated_audio' in locals() and generated_audio and os.path.exists(generated_audio):
        print(f"  🗣️ Custom Phrase: {Path(generated_audio).name}")
    
    print(f"\n💡 Next Steps:")
    print(f"  1. Review generated samples in the project directory")
    print(f"  2. Check the pipeline log for detailed information")
    print(f"  3. For actual fine-tuning, refer to F5-TTS documentation")
    print(f"  4. Experiment with different time ranges or parameters")

if __name__ == "__main__":
    main()
