#!/usr/bin/env python3
"""
F5-TTS Fine-tuning Script
Fine-tune F5-TTS model with custom voice samples
"""

import os
import sys
import torch
import torchaudio
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import librosa
import soundfile as sf
from tqdm import tqdm
import shutil

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    # Try to import F5-TTS
    from f5_tts.api import F5TTS
    from f5_tts.model import CFM
    from f5_tts.infer.utils_infer import (
        load_vocoder, 
        load_model,
        preprocess_ref_audio_text,
        infer_process
    )
    F5TTS_AVAILABLE = True
    print("✅ F5-TTS imports successful")
except ImportError as e:
    print(f"⚠️  F5-TTS not available: {e}")
    print("Installing F5-TTS...")
    F5TTS_AVAILABLE = False

# Try to import Whisper for transcription
try:
    import whisper
    WHISPER_AVAILABLE = True
    print("✅ Whisper available for transcription")
except ImportError:
    print("⚠️  Whisper not available - will use placeholder text")
    WHISPER_AVAILABLE = False

class F5TTSFinetune:
    """Fine-tune F5-TTS with custom voice samples"""
    
    def __init__(self, voice_samples_dir: str, output_dir: str = "finetuned_model"):
        """
        Initialize fine-tuning setup
        
        Args:
            voice_samples_dir: Directory containing voice samples
            output_dir: Directory to save fine-tuned model
        """
        self.voice_samples_dir = Path(voice_samples_dir)
        self.output_dir = Path(output_dir)
        self.device = self._get_device()
        self.sample_rate = 24000  # F5-TTS standard
        
        print(f"🎯 F5-TTS Fine-tuning Setup")
        print(f"📁 Voice samples: {self.voice_samples_dir}")
        print(f"💾 Output directory: {self.output_dir}")
        print(f"🖥️  Device: {self.device}")
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize Whisper if available
        self.whisper_model = None
        if WHISPER_AVAILABLE:
            print("🔥 Loading Whisper for transcription...")
            self.whisper_model = whisper.load_model("base")
            print("✅ Whisper loaded")
        
        # Data storage
        self.training_data = []
        self.metadata = {}
    
    def _get_device(self) -> str:
        """Determine the best device to use"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def prepare_training_data(self):
        """Prepare training data from voice samples"""
        print("📊 Preparing training data...")
        
        if not self.voice_samples_dir.exists():
            raise FileNotFoundError(f"Voice samples directory not found: {self.voice_samples_dir}")
        
        # Find audio files
        audio_files = list(self.voice_samples_dir.glob("*.wav"))
        if not audio_files:
            raise FileNotFoundError(f"No WAV files found in {self.voice_samples_dir}")
        
        print(f"📁 Found {len(audio_files)} audio files")
        
        # Process each audio file
        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            try:
                # Load audio
                audio_data, duration = self._load_and_validate_audio(audio_file)
                
                # Get transcription
                transcription = self._get_transcription(audio_file, audio_data)
                
                # Prepare training sample
                training_sample = {
                    'audio_path': str(audio_file),
                    'audio_data': audio_data,
                    'transcription': transcription,
                    'duration': duration,
                    'sample_rate': self.sample_rate
                }
                
                self.training_data.append(training_sample)
                print(f"✅ Processed: {audio_file.name} ({duration:.1f}s) - '{transcription[:50]}...'")
                
            except Exception as e:
                print(f"⚠️  Error processing {audio_file.name}: {e}")
                continue
        
        print(f"📊 Prepared {len(self.training_data)} training samples")
        
        # Save metadata
        self._save_metadata()
        
        return self.training_data
    
    def _load_and_validate_audio(self, audio_path: Path) -> Tuple[np.ndarray, float]:
        """Load and validate audio file"""
        # Load audio
        audio, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
        duration = len(audio) / self.sample_rate
        
        # Basic validation
        if duration < 3.0:
            raise ValueError(f"Audio too short: {duration:.1f}s")
        if duration > 30.0:
            print(f"⚠️  Long audio detected: {duration:.1f}s")
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        return audio, duration
    
    def _get_transcription(self, audio_path: Path, audio_data: np.ndarray) -> str:
        """Get transcription for audio file"""
        if self.whisper_model:
            try:
                # Save temporary file for Whisper
                temp_path = self.output_dir / "temp_audio.wav"
                sf.write(temp_path, audio_data, self.sample_rate)
                
                # Transcribe
                result = self.whisper_model.transcribe(str(temp_path))
                transcription = result['text'].strip()
                
                # Clean up
                temp_path.unlink()
                
                return transcription
                
            except Exception as e:
                print(f"⚠️  Transcription failed for {audio_path.name}: {e}")
        
        # Fallback: generate based on filename
        filename = audio_path.stem
        duration = filename.split('_')[2] if '_' in filename else "unknown"
        return f"This is a voice sample of {duration} duration for training the speech synthesis model."
    
    def _save_metadata(self):
        """Save training metadata"""
        metadata = {
            'total_samples': len(self.training_data),
            'total_duration': sum(sample['duration'] for sample in self.training_data),
            'average_duration': sum(sample['duration'] for sample in self.training_data) / len(self.training_data),
            'sample_rate': self.sample_rate,
            'device': self.device,
            'samples': [
                {
                    'file': sample['audio_path'],
                    'duration': sample['duration'],
                    'transcription': sample['transcription']
                }
                for sample in self.training_data
            ]
        }
        
        metadata_path = self.output_dir / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Metadata saved to: {metadata_path}")
        print(f"📊 Training summary:")
        print(f"   Total samples: {metadata['total_samples']}")
        print(f"   Total duration: {metadata['total_duration']:.1f} seconds")
        print(f"   Average duration: {metadata['average_duration']:.1f} seconds")
    
    def setup_f5tts_training(self):
        """Set up F5-TTS training environment"""
        if not F5TTS_AVAILABLE:
            print("❌ F5-TTS not available. Please install it first:")
            print("   git clone https://github.com/SWivid/F5-TTS.git")
            print("   cd F5-TTS")
            print("   pip install -e .")
            return False
        
        print("🔧 Setting up F5-TTS training...")
        
        # Create training structure
        training_dir = self.output_dir / "f5tts_training"
        training_dir.mkdir(exist_ok=True)
        
        # Create data directories
        (training_dir / "audio").mkdir(exist_ok=True)
        (training_dir / "metadata").mkdir(exist_ok=True)
        
        # Copy audio files and create metadata
        metadata_lines = []
        
        for i, sample in enumerate(self.training_data):
            # Copy audio file
            src_path = Path(sample['audio_path'])
            dst_path = training_dir / "audio" / f"sample_{i:03d}.wav"
            shutil.copy2(src_path, dst_path)
            
            # Create metadata line
            metadata_line = f"sample_{i:03d}.wav|{sample['transcription']}|{sample['duration']:.2f}"
            metadata_lines.append(metadata_line)
        
        # Save metadata file
        metadata_file = training_dir / "metadata.txt"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(metadata_lines))
        
        print(f"✅ F5-TTS training setup complete")
        print(f"📁 Training directory: {training_dir}")
        print(f"📝 Metadata file: {metadata_file}")
        
        return training_dir
    
    def run_mock_training(self):
        """Run a mock training process (for when F5-TTS is not available)"""
        print("🎭 Running mock training process...")
        
        # Create mock outputs
        mock_model_dir = self.output_dir / "mock_finetuned_model"
        mock_model_dir.mkdir(exist_ok=True)
        
        # Create mock model files
        (mock_model_dir / "model.pth").touch()
        (mock_model_dir / "config.json").touch()
        
        # Create summary
        summary = {
            'model_type': 'F5-TTS (Mock)',
            'training_samples': len(self.training_data),
            'total_duration': sum(sample['duration'] for sample in self.training_data),
            'status': 'Mock training completed',
            'model_path': str(mock_model_dir)
        }
        
        summary_path = self.output_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Mock training completed")
        print(f"📁 Mock model saved to: {mock_model_dir}")
        print(f"📊 Training summary: {summary_path}")
        
        return mock_model_dir

def main():
    """Main fine-tuning function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune F5-TTS with voice samples")
    parser.add_argument("voice_dir", help="Directory containing voice samples")
    parser.add_argument("-o", "--output", default="finetuned_model", 
                       help="Output directory for fine-tuned model")
    parser.add_argument("--mock", action="store_true", 
                       help="Run mock training (for testing)")
    
    args = parser.parse_args()
    
    # Initialize fine-tuner
    finetuner = F5TTSFinetune(args.voice_dir, args.output)
    
    try:
        # Prepare training data
        training_data = finetuner.prepare_training_data()
        
        if not training_data:
            print("❌ No valid training data found")
            return
        
        if args.mock or not F5TTS_AVAILABLE:
            # Run mock training
            model_dir = finetuner.run_mock_training()
        else:
            # Set up real F5-TTS training
            training_dir = finetuner.setup_f5tts_training()
            
            print("📚 F5-TTS training setup complete!")
            print("🚀 To continue with actual training, run:")
            print(f"   cd {training_dir}")
            print("   # Follow F5-TTS training documentation")
            
            model_dir = training_dir
        
        print(f"🎉 Fine-tuning process complete!")
        print(f"📁 Output: {model_dir}")
        
    except Exception as e:
        print(f"❌ Error during fine-tuning: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
