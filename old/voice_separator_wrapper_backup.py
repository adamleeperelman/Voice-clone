#!/usr/bin/env python3
"""
Voice AI Separator Module
Modular version of the voice separation functionality
Combines AudioProcessor and VoiceSeparator for complete functionality
"""

import os
from typing import Dict, List, Optional
from .audio_processor import AudioProcessor
from .voice_separator_core import VoiceSeparator


class VoiceAISeparator:
    """
    Complete Voice AI Separator
    Combines preprocessing and voice separation functionality
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the Voice AI Separator with both processors"""
        self.audio_processor = AudioProcessor()
        self.voice_separator = VoiceSeparator(openai_api_key)
        print("🤖 Voice AI Separator initialized")
    
    # Audio Processing Methods (delegate to AudioProcessor)
    def extract_time_range(self, input_path: str, start_minutes: float, 
                          end_minutes: float, output_path: str = None) -> str:
        """Extract a specific time range from audio"""
        return self.audio_processor.extract_time_range(input_path, start_minutes, end_minutes, output_path)
    
    def separate_channels(self, input_path: str, output_dir: str = None) -> Dict[str, str]:
        """Separate stereo audio into left and right channels"""
        return self.audio_processor.separate_channels(input_path, output_dir)
    
    def detect_voice_activity(self, input_path: str, min_voice_threshold: float = 0.02, 
                             min_voice_ratio: float = 0.3) -> Dict:
        """Detect voice activity in an audio file"""
        return self.audio_processor.detect_voice_activity(input_path, min_voice_threshold, min_voice_ratio)
    
    def filter_audio_by_voice_activity(self, input_paths: List[str], output_dir: str = None,
                                      min_voice_threshold: float = 0.02, 
                                      min_voice_ratio: float = 0.3) -> List[str]:
        """Filter audio files by voice activity"""
        return self.audio_processor.filter_audio_by_voice_activity(
            input_paths, output_dir, min_voice_threshold, min_voice_ratio)
    
    # Voice Separation Methods (delegate to VoiceSeparator)
    def load_whisper_model(self, model_size: str = "base"):
        """Load Whisper model for transcription"""
        return self.voice_separator.load_whisper_model(model_size)
    
    def separate_audio(self, input_path: str, output_dir: str = "ai_voice_samples",
                      min_duration: float = 8.0, min_confidence: float = -0.5,
                      silence_len: int = 2000, silence_thresh: int = -35, 
                      min_segment_len: float = 8.0, max_segment_len: float = 15.0,
                      max_no_speech_prob: float = 0.3) -> List[str]:
        """Main method to separate audio into speakers with quality filtering"""
        return self.voice_separator.separate_audio(
            input_path, output_dir, min_duration, min_confidence,
            silence_len, silence_thresh, min_segment_len, max_segment_len, max_no_speech_prob
        )
