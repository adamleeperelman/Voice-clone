#!/usr/bin/env python3
"""
Voice AI Module
Modular voice processing system with separation, training, and synthesis
"""

# Import all components for easy access
from .audio_processor import AudioProcessor
from .voice_separator import VoiceSeparator
from .voice_trainer import VoiceTrainer
from .voice_synthesizer import VoiceSynthesizer
from .processor import VoiceAIProcessor, create_processor

# Make all classes available at module level
__all__ = [
    'VoiceAIProcessor',
    'AudioProcessor',
    'VoiceSeparator', 
    'VoiceTrainer',
    'VoiceSynthesizer',
    'create_processor'
]
