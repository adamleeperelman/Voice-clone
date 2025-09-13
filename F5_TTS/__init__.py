"""
F5-TTS Voice Cloning Implementation
Loads voice samples and provides text-to-speech with cloned voices
"""

from .model import F5TTSModel
from .voice_cloner import VoiceCloner

__version__ = "1.0.0"
__all__ = ["F5TTSModel", "VoiceCloner"]
