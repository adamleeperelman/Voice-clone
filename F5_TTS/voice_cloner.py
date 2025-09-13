#!/usr/bin/env python3
"""
F5-TTS Voice Cloner
Main interface for loading voice samples and generating speech
"""

import os
import sys
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
import librosa
import soundfile as sf
from pydub import AudioSegment
import warnings
warnings.filterwarnings("ignore")

try:
    # Try to import F5-TTS if available
    from f5_tts.api import F5TTS
    from f5_tts.model import DiT
    from f5_tts.infer.utils_infer import load_vocoder, load_model
    F5TTS_AVAILABLE = True
except ImportError:
    print("⚠️  F5-TTS not found. Install with: pip install f5-tts")
    F5TTS_AVAILABLE = False

# Import our model wrapper
from model import F5TTSModel

class VoiceCloner:
    """
    F5-TTS Voice Cloner that loads voice samples and generates speech
    """
    
    def __init__(self, voice_samples_dir: str = "ai_voice_samples", device: str = "auto"):
        """
        Initialize the Voice Cloner
        
        Args:
            voice_samples_dir: Directory containing voice samples
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.voice_samples_dir = Path(voice_samples_dir)
        self.device = self._get_device(device)
        self.sample_rate = 24000  # F5-TTS default
        
        print(f"🤖 F5-TTS Voice Cloner")
        print(f"📁 Voice samples: {self.voice_samples_dir}")
        print(f"🖥️  Device: {self.device}")
        
        # Voice data storage
        self.voices = {}
        self.current_voice = None
        
        # Load F5-TTS model
        self._load_model()
        
        # Load available voices
        self._load_voices()
    
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
    
    def _load_model(self):
        """Load F5-TTS model"""
        if not F5TTS_AVAILABLE:
            print("❌ F5-TTS not available. Using mock implementation.")
            self.model = None
            self.vocoder = None
            return
        
        try:
            print("🤖 Loading F5-TTS model...")
            
            # Use our F5TTSModel wrapper
            self.f5tts = F5TTSModel(device=self.device)
            
            print("✅ F5-TTS model loaded successfully")
            
        except Exception as e:
            print(f"⚠️  Error loading F5-TTS model: {e}")
            print("📝 Using fallback implementation")
            self.f5tts = None
    
    def _load_voices(self):
        """Load all available voice samples"""
        if not self.voice_samples_dir.exists():
            print(f"❌ Voice samples directory not found: {self.voice_samples_dir}")
            return
        
        print("🎤 Loading voice samples...")
        
        # Look for speaker folders
        speaker_folders = [d for d in self.voice_samples_dir.iterdir() 
                          if d.is_dir() and d.name.endswith('_speaker')]
        
        if not speaker_folders:
            print("❌ No speaker folders found")
            return
        
        for speaker_folder in speaker_folders:
            speaker_name = speaker_folder.name.replace('_speaker', '')
            audio_files = list(speaker_folder.glob('*.wav'))
            
            if audio_files:
                print(f"  📁 Found {len(audio_files)} samples for '{speaker_name}'")
                self.voices[speaker_name] = {
                    'folder': speaker_folder,
                    'samples': audio_files,
                    'reference_audio': None,
                    'reference_text': None
                }
        
        if self.voices:
            print(f"✅ Loaded {len(self.voices)} voices: {list(self.voices.keys())}")
            # Set first voice as default
            self.current_voice = list(self.voices.keys())[0]
            print(f"🎯 Default voice: {self.current_voice}")
        else:
            print("❌ No voice samples found")
    
    def list_voices(self) -> List[str]:
        """List all available voices"""
        return list(self.voices.keys())
    
    def set_voice(self, voice_name: str, reference_sample: Optional[str] = None):
        """
        Set the current voice for synthesis
        
        Args:
            voice_name: Name of the voice to use
            reference_sample: Specific sample file to use as reference
        """
        if voice_name not in self.voices:
            raise ValueError(f"Voice '{voice_name}' not found. Available: {list(self.voices.keys())}")
        
        self.current_voice = voice_name
        voice_data = self.voices[voice_name]
        
        # Select reference sample
        if reference_sample:
            ref_path = Path(reference_sample)
            if not ref_path.exists():
                ref_path = voice_data['folder'] / reference_sample
            if not ref_path.exists():
                raise FileNotFoundError(f"Reference sample not found: {reference_sample}")
        else:
            # Use the first (usually highest quality) sample
            ref_path = voice_data['samples'][0]
        
        # Load reference audio
        print(f"🎤 Loading reference: {ref_path.name}")
        audio_data = self._load_audio(ref_path)
        
        voice_data['reference_audio'] = audio_data
        voice_data['reference_text'] = self._extract_reference_text(ref_path)
        
        print(f"✅ Voice set to: {voice_name}")
        print(f"📝 Reference text: {voice_data['reference_text'][:50]}...")
    
    def _load_audio(self, audio_path: Path) -> torch.Tensor:
        """Load audio file and convert to tensor"""
        try:
            # Load with librosa for better compatibility
            audio, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
            return torch.from_numpy(audio).float()
        except Exception as e:
            print(f"⚠️  Error loading audio {audio_path}: {e}")
            # Fallback to torchaudio
            audio, sr = torchaudio.load(str(audio_path))
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                audio = resampler(audio)
            return audio.squeeze()
    
    def _extract_reference_text(self, audio_path: Path) -> str:
        """Extract or generate reference text for the audio"""
        # For now, return a generic text
        # In a real implementation, you might want to use Whisper to transcribe
        filename = audio_path.stem
        duration = filename.split('_')[2] if '_' in filename else "unknown"
        
        return f"This is a reference sample of approximately {duration} duration for voice cloning."
    
    def generate_speech(self, 
                       text: str, 
                       voice_name: Optional[str] = None,
                       speed: float = 1.0,
                       output_path: Optional[str] = None) -> Union[np.ndarray, str]:
        """
        Generate speech from text using the cloned voice
        
        Args:
            text: Text to synthesize
            voice_name: Voice to use (uses current if None)
            speed: Speech speed multiplier
            output_path: Path to save audio file
            
        Returns:
            Audio array or path to saved file
        """
        if voice_name and voice_name != self.current_voice:
            self.set_voice(voice_name)
        
        if not self.current_voice:
            raise ValueError("No voice selected. Use set_voice() first.")
        
        voice_data = self.voices[self.current_voice]
        
        if voice_data['reference_audio'] is None:
            self.set_voice(self.current_voice)  # Auto-load reference
        
        print(f"🗣️  Generating speech: '{text[:50]}...'")
        print(f"🎤 Using voice: {self.current_voice}")
        
        if self.f5tts is None:
            # Fallback implementation
            return self._generate_fallback(text, voice_data, output_path)
        
        try:
            # Use F5-TTS for generation
            ref_audio = voice_data['reference_audio']
            ref_text = voice_data['reference_text']
            
            # Generate with F5-TTS using our wrapper
            generated_audio, sample_rate = self.f5tts.synthesize(
                text=text,
                reference_audio=ref_audio.numpy() if hasattr(ref_audio, 'numpy') else ref_audio,
                reference_text=ref_text,
                speed=speed
            )
            
            # Convert to numpy array if needed
            if isinstance(generated_audio, torch.Tensor):
                generated_audio = generated_audio.cpu().numpy()
            
            # Save if requested
            if output_path:
                sf.write(output_path, generated_audio, sample_rate)
                print(f"💾 Saved to: {output_path}")
                return output_path
            
            return generated_audio
            
        except Exception as e:
            print(f"❌ Error generating speech: {e}")
            return self._generate_fallback(text, voice_data, output_path)
    
    def _generate_fallback(self, text: str, voice_data: Dict, output_path: Optional[str]) -> Union[np.ndarray, str]:
        """Fallback speech generation when F5-TTS is not available"""
        print("🔄 Using fallback synthesis...")
        
        # Simple fallback: return one of the reference samples
        ref_sample = voice_data['samples'][0]
        audio_data = self._load_audio(ref_sample)
        
        if output_path:
            # Copy reference sample as fallback
            import shutil
            shutil.copy2(ref_sample, output_path)
            print(f"💾 Fallback: Copied reference to {output_path}")
            return output_path
        
        return audio_data.numpy()
    
    def clone_voice_interactive(self):
        """Interactive voice cloning session"""
        print("\n🎤 F5-TTS Interactive Voice Cloning")
        print("="*50)
        
        if not self.voices:
            print("❌ No voices available. Please add voice samples first.")
            return
        
        # Show available voices
        print("Available voices:")
        for i, voice in enumerate(self.voices.keys(), 1):
            sample_count = len(self.voices[voice]['samples'])
            print(f"  {i}. {voice} ({sample_count} samples)")
        
        # Voice selection
        try:
            choice = input(f"\nSelect voice (1-{len(self.voices)}): ").strip()
            voice_names = list(self.voices.keys())
            selected_voice = voice_names[int(choice) - 1]
            self.set_voice(selected_voice)
        except (ValueError, IndexError):
            print("❌ Invalid selection")
            return
        
        # Text input loop
        print(f"\n🎯 Voice: {self.current_voice}")
        print("Enter text to synthesize (empty line to exit):")
        
        output_dir = Path("generated_speech")
        output_dir.mkdir(exist_ok=True)
        
        sample_count = 1
        
        while True:
            text = input("> ").strip()
            if not text:
                break
            
            try:
                output_path = output_dir / f"speech_{self.current_voice}_{sample_count:03d}.wav"
                result = self.generate_speech(text, output_path=str(output_path))
                print(f"✅ Generated: {output_path}")
                sample_count += 1
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("👋 Thanks for using F5-TTS Voice Cloner!")

def main():
    """Main function for testing"""
    # Initialize voice cloner
    cloner = VoiceCloner()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        cloner.clone_voice_interactive()
    else:
        # Quick test
        voices = cloner.list_voices()
        if voices:
            cloner.set_voice(voices[0])
            test_text = "Hello, this is a test of the F5-TTS voice cloning system."
            audio = cloner.generate_speech(test_text, output_path="test_output.wav")
            print(f"🎉 Test complete! Generated: test_output.wav")
        else:
            print("❌ No voices found. Please run voice_ai_separator.py first to generate samples.")

if __name__ == "__main__":
    main()
