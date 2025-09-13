#!/usr/bin/env python3
"""
F5-TTS Model Wrapper
Provides a simplified interface for F5-TTS model operations
"""

import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
import tempfile
import os
from typing import Optional, Tuple, Union
from pathlib import Path


class F5TTSModel:
    """
    Wrapper for F5-TTS model with simplified interface
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = "auto",
                 sample_rate: int = 24000):
        """
        Initialize F5-TTS Model
        
        Args:
            model_path: Path to model checkpoint
            device: Device to use
            sample_rate: Audio sample rate
        """
        self.device = self._get_device(device)
        self.sample_rate = sample_rate
        self.model = None
        self.vocoder = None
        
        print(f"🤖 Initializing F5-TTS Model")
        print(f"🖥️  Device: {self.device}")
        print(f"🎵 Sample Rate: {self.sample_rate}")
        
        # Try to load the actual model
        self._load_model(model_path)
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_model(self, model_path: Optional[str]):
        """Load the F5-TTS model"""
        try:
            # Try to import and load F5-TTS
            from f5_tts.api import F5TTS
            
            if model_path:
                print(f"📁 Loading model from: {model_path}")
            else:
                print("📁 Loading pretrained F5-TTS model...")
            
            self.model = F5TTS(
                model="F5TTS_v1_Base",
                ckpt_file=model_path or "",
                device=self.device
            )
            
            print("✅ F5-TTS model loaded successfully")
            
        except ImportError:
            print("⚠️  F5-TTS not installed. Install with:")
            print("   pip install f5-tts")
            print("📝 Using mock model for testing")
            self.model = MockF5TTSModel(self.device)
            
        except Exception as e:
            print(f"⚠️  Error loading F5-TTS: {e}")
            print("📝 Using mock model for testing")
            self.model = MockF5TTSModel(self.device)
    
    def synthesize(self, 
                   text: str,
                   reference_audio: Union[torch.Tensor, np.ndarray],
                   reference_text: str,
                   speed: float = 1.0) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from text using reference voice
        
        Args:
            text: Text to synthesize
            reference_audio: Reference audio for voice cloning
            reference_text: Reference text corresponding to reference audio
            speed: Speech speed multiplier
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Convert reference audio to numpy if needed
        if isinstance(reference_audio, torch.Tensor):
            reference_audio = reference_audio.cpu().numpy()
        
        print(f"🗣️  Synthesizing: '{text[:50]}...'")
        
        try:
            # Use the model to generate speech
            if hasattr(self.model, 'infer'):
                # Real F5-TTS model - needs file path for reference
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    # Save reference audio to temporary file
                    sf.write(temp_file.name, reference_audio, self.sample_rate)
                    temp_ref_path = temp_file.name
                
                try:
                    # Call F5-TTS infer with file path
                    result = self.model.infer(
                        ref_file=temp_ref_path,
                        ref_text=reference_text,
                        gen_text=text,
                        speed=speed,
                        show_info=lambda x: None,  # Suppress verbose output
                        remove_silence=True
                    )
                    
                    # F5-TTS returns (audio_array, sample_rate, spectrogram)
                    if isinstance(result, tuple) and len(result) >= 2:
                        audio, sr = result[0], result[1]
                    else:
                        raise RuntimeError(f"Unexpected F5-TTS output format: {type(result)}")
                        
                finally:
                    # Clean up temporary reference file
                    try:
                        os.unlink(temp_ref_path)
                    except:
                        pass
                        
            else:
                # Mock model
                audio, sr = self.model.generate(
                    text=text,
                    reference_audio=reference_audio,
                    reference_text=reference_text,
                    speed=speed
                )
            
            return audio, sr
            
        except Exception as e:
            print(f"❌ Synthesis error: {e}")
            # Return silence as fallback
            duration = max(1.0, len(text) * 0.1)  # Rough estimate
            silence = np.zeros(int(duration * self.sample_rate))
            return silence, self.sample_rate
    
    def is_available(self) -> bool:
        """Check if the model is available and ready"""
        return self.model is not None


class MockF5TTSModel:
    """
    Mock F5-TTS model for testing when the real model is not available
    """
    
    def __init__(self, device: str):
        self.device = device
        print("🔧 Using mock F5-TTS model for testing")
    
    def generate(self, 
                text: str,
                reference_audio: np.ndarray,
                reference_text: str,
                speed: float = 1.0) -> Tuple[np.ndarray, int]:
        """
        Mock speech generation
        """
        print(f"🔄 Mock generation: '{text[:30]}...'")
        
        # Generate simple sine wave as placeholder
        sample_rate = 24000
        duration = max(1.0, len(text) * 0.08 / speed)  # Rough estimate
        
        # Create a simple synthesized sound
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Mix of different frequencies to simulate speech
        freq1 = 200 + len(text) % 100  # Base frequency
        freq2 = 400 + (len(text) * 7) % 200  # Harmonic
        
        audio = (
            0.3 * np.sin(2 * np.pi * freq1 * t) +
            0.2 * np.sin(2 * np.pi * freq2 * t) +
            0.1 * np.random.normal(0, 0.1, len(t))  # Add some noise
        )
        
        # Apply envelope to make it more speech-like
        envelope = np.exp(-t * 0.5) * (1 - np.exp(-t * 5))
        audio *= envelope
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.7
        
        return audio, sample_rate


# Convenience function
def create_model(model_path: Optional[str] = None, device: str = "auto") -> F5TTSModel:
    """
    Create and return an F5-TTS model instance
    
    Args:
        model_path: Path to model checkpoint (optional)
        device: Device to use ('cpu', 'cuda', 'mps', or 'auto')
        
    Returns:
        F5TTSModel instance
    """
    return F5TTSModel(model_path=model_path, device=device)
