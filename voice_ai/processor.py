#!/usr/bin/env python3
"""
Voice AI Processor
Main orchestrator class for voice separation, training, and synthesis
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

# Import the component modules
from .audio_processor import AudioProcessor
from .voice_separator import VoiceSeparator
from .voice_trainer import VoiceTrainer
from .voice_synthesizer import VoiceSynthesizer

class VoiceAIProcessor:
    """
    Main Voice AI Processor
    Unified interface for all voice processing operations
    """
    
    def __init__(self, workspace_path: str = None, openai_api_key: str = None):
        """Initialize the Voice AI Processor with all components"""
        self.workspace_path = workspace_path or "."
        
        # Initialize components directly
        self.audio_processor = AudioProcessor()
        self.voice_separator = VoiceSeparator(openai_api_key)
        self.trainer = VoiceTrainer()
        self.synthesizer = VoiceSynthesizer()
        
        print("🚀 Voice AI Processor initialized")
        print(f"📁 Workspace: {self.workspace_path}")
    
    # Voice Separation Methods
    def separate_voices(self, 
                       input_path: str, 
                       output_dir: str = "ai_voice_samples",
                       **kwargs) -> List[str]:
        """
        Separate audio into different speakers
        
        Args:
            input_path: Path to input audio file
            output_dir: Directory to save separated audio
            **kwargs: Additional parameters for separation
        
        Returns:
            List of saved audio file paths
        """
        return self.voice_separator.separate_audio(input_path, output_dir, **kwargs)
    
    def extract_time_range(self, 
                          input_path: str, 
                          start_minutes: float, 
                          end_minutes: float, 
                          output_path: str = None) -> str:
        """
        Extract a specific time range from an audio file
        
        Args:
            input_path: Path to the input audio file
            start_minutes: Start time in minutes (e.g., 0, 5.5, 10)
            end_minutes: End time in minutes (e.g., 15, 30.5, 60)
            output_path: Optional output path, auto-generated if not provided
        
        Returns:
            Path to the extracted audio file
        """
        return self.audio_processor.extract_time_range(input_path, start_minutes, end_minutes, output_path)
    
    def separate_channels(self, input_path: str, output_dir: str = None) -> Dict[str, str]:
        """
        Separate stereo audio into left and right channels
        
        Args:
            input_path: Path to stereo input audio file
            output_dir: Directory to save channel files (auto-generated if not provided)
        
        Returns:
            Dictionary with 'left' and 'right' keys containing file paths
        """
        return self.audio_processor.separate_channels(input_path, output_dir)
    
    def detect_voice_activity(self, input_path: str, min_voice_threshold: float = 0.02, 
                             min_voice_ratio: float = 0.3) -> Dict:
        """
        Detect voice activity in an audio file
        
        Args:
            input_path: Path to audio file to analyze
            min_voice_threshold: Minimum RMS energy to consider as voice
            min_voice_ratio: Minimum ratio of voice frames to total frames
        
        Returns:
            Dictionary with voice activity analysis results
        """
        return self.audio_processor.detect_voice_activity(input_path, min_voice_threshold, min_voice_ratio)
    
    def filter_audio_by_voice_activity(self, input_paths: List[str], output_dir: str = None,
                                      min_voice_threshold: float = 0.02, 
                                      min_voice_ratio: float = 0.3) -> List[str]:
        """
        Filter audio files to keep only those with sufficient voice activity
        
        Args:
            input_paths: List of audio file paths to filter
            output_dir: Directory to copy filtered files (optional)
            min_voice_threshold: Minimum RMS energy threshold
            min_voice_ratio: Minimum ratio of voice frames
        
        Returns:
            List of paths to files with sufficient voice activity
        """
        return self.audio_processor.filter_audio_by_voice_activity(
            input_paths, output_dir, min_voice_threshold, min_voice_ratio)
    
    # Training Methods
    def prepare_training_data(self, 
                            source_dir: str, 
                            output_dir: str = "F5_TTS/finetune_data",
                            speaker_name: str = "custom_speaker") -> Dict:
        """
        Prepare training data for voice cloning
        
        Args:
            source_dir: Directory containing audio samples
            output_dir: Directory to save training data
            speaker_name: Name for the speaker
        
        Returns:
            Training data metadata
        """
        return self.trainer.prepare_training_data(source_dir, output_dir, speaker_name)
    
    def fine_tune_model(self, 
                       training_dir: str = "F5_TTS/finetune_data",
                       **kwargs) -> Dict:
        """
        Fine-tune voice model
        
        Args:
            training_dir: Directory containing training data
            **kwargs: Additional training parameters
        
        Returns:
            Training results
        """
        return self.trainer.fine_tune_model(training_dir, **kwargs)
    
    def validate_training_data(self, training_dir: str = "F5_TTS/finetune_data") -> Dict:
        """
        Validate training data for compatibility
        
        Args:
            training_dir: Directory containing training data
        
        Returns:
            Validation results
        """
        return self.trainer.validate_training_data(training_dir)
    
    # Synthesis Methods
    def synthesize_speech(self, 
                         text: str, 
                         reference_audio: str = None,
                         model: str = "F5-TTS",
                         output_path: str = None,
                         **kwargs) -> str:
        """
        Generate speech from text
        
        Args:
            text: Text to synthesize
            reference_audio: Reference audio for voice cloning
            model: Model to use for synthesis
            output_path: Output file path
            **kwargs: Additional synthesis parameters
        
        Returns:
            Path to generated audio file
        """
        return self.synthesizer.generate_audio(text, reference_audio, model, output_path, **kwargs)
    
    def batch_synthesize(self, 
                        texts: List[str], 
                        reference_audio: str = None,
                        **kwargs) -> List[str]:
        """
        Generate multiple audio files from text list
        
        Args:
            texts: List of texts to synthesize
            reference_audio: Reference audio for voice cloning
            **kwargs: Additional synthesis parameters
        
        Returns:
            List of generated audio file paths
        """
        return self.synthesizer.batch_generate(texts, reference_audio, **kwargs)
    
    def test_voice_cloning(self, 
                          reference_audio: str = None,
                          **kwargs) -> List[str]:
        """
        Run voice cloning tests
        
        Args:
            reference_audio: Reference audio for voice cloning
            **kwargs: Additional test parameters
        
        Returns:
            List of test audio file paths
        """
        return self.synthesizer.test_voice_cloning(reference_audio, **kwargs)
    
    def synthesize_from_file(self, 
                           text_file: str,
                           reference_audio: str = None,
                           **kwargs) -> List[str]:
        """
        Synthesize audio from text file
        
        Args:
            text_file: Path to text file
            reference_audio: Reference audio for voice cloning
            **kwargs: Additional synthesis parameters
        
        Returns:
            List of generated audio file paths
        """
        return self.synthesizer.synthesize_from_file(text_file, reference_audio, **kwargs)
    
    # Pipeline Methods
    def full_pipeline(self, 
                     input_audio: str,
                     test_text: str = "Hello, this is a test of the voice cloning system.",
                     speaker_name: str = "custom_speaker") -> Dict:
        """
        Run the complete voice cloning pipeline
        
        Args:
            input_audio: Path to input audio file
            test_text: Text to synthesize for testing
            speaker_name: Name for the speaker
        
        Returns:
            Results from each step of the pipeline
        """
        results = {
            "separation": {},
            "training": {},
            "synthesis": {},
            "success": False
        }
        
        try:
            # Step 1: Separate voices
            print("\n🎯 Step 1: Voice Separation")
            separated_files = self.separate_voices(input_audio)
            results["separation"]["files"] = separated_files
            results["separation"]["success"] = len(separated_files) > 0
            
            if not separated_files:
                results["error"] = "Voice separation failed"
                return results
            
            # Step 2: Prepare training data
            print("\n🎯 Step 2: Training Data Preparation")
            left_speaker_dir = os.path.join(self.workspace_path, "ai_voice_samples", "left_speaker")
            training_metadata = self.prepare_training_data(left_speaker_dir, speaker_name=speaker_name)
            results["training"]["metadata"] = training_metadata
            results["training"]["success"] = bool(training_metadata)
            
            if not training_metadata:
                results["error"] = "Training data preparation failed"
                return results
            
            # Step 3: Test voice synthesis
            print("\n🎯 Step 3: Voice Synthesis Test")
            test_files = self.test_voice_cloning()
            results["synthesis"]["test_files"] = test_files
            results["synthesis"]["success"] = len(test_files) > 0
            
            # Step 4: Custom text synthesis
            if test_text:
                print("\n🎯 Step 4: Custom Text Synthesis")
                custom_audio = self.synthesize_speech(test_text)
                results["synthesis"]["custom_audio"] = custom_audio
            
            results["success"] = all([
                results["separation"]["success"],
                results["training"]["success"],
                results["synthesis"]["success"]
            ])
            
        except Exception as e:
            results["error"] = f"Pipeline error: {e}"
        
        return results
    
    # Status and Utility Methods
    def get_workspace_status(self) -> Dict:
        """
        Get status of entire voice processing workspace
        
        Returns:
            Comprehensive status information
        """
        status = {
            "workspace_path": self.workspace_path,
            "separation": {},
            "training": {},
            "synthesis": {}
        }
        
        # Get separation status
        try:
            # Check for separated audio
            samples_dir = Path(self.workspace_path) / "ai_voice_samples"
            if samples_dir.exists():
                left_speaker = samples_dir / "left_speaker"
                right_speaker = samples_dir / "right_speaker"
                status["separation"] = {
                    "samples_directory_exists": True,
                    "left_speaker_files": len(list(left_speaker.glob("*.wav"))) if left_speaker.exists() else 0,
                    "right_speaker_files": len(list(right_speaker.glob("*.wav"))) if right_speaker.exists() else 0
                }
            else:
                status["separation"]["samples_directory_exists"] = False
        except Exception as e:
            status["separation"]["error"] = str(e)
        
        # Get training status
        try:
            status["training"] = self.trainer.get_training_status()
        except Exception as e:
            status["training"]["error"] = str(e)
        
        # Get synthesis status
        try:
            status["synthesis"] = self.synthesizer.get_synthesis_status()
        except Exception as e:
            status["synthesis"]["error"] = str(e)
        
        return status
    
    def get_available_models(self) -> List[str]:
        """Get list of available synthesis models"""
        return self.synthesizer.get_available_models()
    
    def find_reference_audio(self, speaker_dir: str = None) -> Optional[str]:
        """Find suitable reference audio for synthesis"""
        return self.synthesizer.find_reference_audio(speaker_dir)


# Convenience function for quick access
def create_processor(workspace_path: str = None, openai_api_key: str = None) -> VoiceAIProcessor:
    """
    Create a VoiceAIProcessor instance with simplified interface
    
    Args:
        workspace_path: Path to workspace directory
        openai_api_key: OpenAI API key for enhanced separation
    
    Returns:
        Configured VoiceAIProcessor instance
    """
    return VoiceAIProcessor(workspace_path=workspace_path, openai_api_key=openai_api_key)
