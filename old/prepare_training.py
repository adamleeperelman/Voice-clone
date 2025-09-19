#!/usr/bin/env python3
"""
Simple Voice Cloning Training Script
Prepares data for F5-TTS training with the left_speaker samples
"""

import os
import sys
import json
import shutil
from pathlib import Path
import librosa
import soundfile as sf
from typing import List, Dict

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

# Try to import whisper for transcription
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️  Whisper not available - using placeholder transcriptions")

class SimpleVoiceTrainer:
    """Simple voice training data preparation"""
    
    def __init__(self, samples_dir: str, output_dir: str = "training_data"):
        self.samples_dir = Path(samples_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load Whisper if available
        self.whisper_model = None
        if WHISPER_AVAILABLE:
            print("🔥 Loading Whisper for transcription...")
            self.whisper_model = whisper.load_model("base")
    
    def transcribe_audio(self, audio_path: Path) -> str:
        """Transcribe audio file"""
        if self.whisper_model:
            try:
                result = self.whisper_model.transcribe(str(audio_path))
                return result['text'].strip()
            except Exception as e:
                print(f"⚠️  Transcription failed for {audio_path.name}: {e}")
        
        # Fallback text based on filename
        filename = audio_path.stem
        return f"This is a voice sample for training the speech synthesis model."
    
    def prepare_training_data(self) -> Dict:
        """Prepare training data from voice samples"""
        print(f"📊 Preparing training data from: {self.samples_dir}")
        
        if not self.samples_dir.exists():
            raise FileNotFoundError(f"Samples directory not found: {self.samples_dir}")
        
        # Find audio files
        audio_files = list(self.samples_dir.glob("*.wav"))
        if not audio_files:
            raise FileNotFoundError(f"No WAV files found in {self.samples_dir}")
        
        print(f"📁 Found {len(audio_files)} audio files")
        
        # Process files
        training_data = []
        total_duration = 0
        
        for audio_file in audio_files:
            try:
                # Load audio to get duration
                audio, sr = librosa.load(str(audio_file), sr=None)
                duration = len(audio) / sr
                total_duration += duration
                
                # Get transcription
                transcription = self.transcribe_audio(audio_file)
                
                # Prepare sample data
                sample_data = {
                    'file': audio_file.name,
                    'path': str(audio_file),
                    'duration': duration,
                    'transcription': transcription,
                    'sample_rate': sr
                }
                
                training_data.append(sample_data)
                print(f"✅ {audio_file.name}: {duration:.1f}s - '{transcription[:50]}...'")
                
            except Exception as e:
                print(f"⚠️  Error processing {audio_file.name}: {e}")
                continue
        
        # Create metadata
        metadata = {
            'total_samples': len(training_data),
            'total_duration': total_duration,
            'average_duration': total_duration / len(training_data) if training_data else 0,
            'samples': training_data
        }
        
        # Save metadata
        metadata_path = self.output_dir / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n📊 Training Data Summary:")
        print(f"   Total samples: {metadata['total_samples']}")
        print(f"   Total duration: {metadata['total_duration']:.1f} seconds")
        print(f"   Average duration: {metadata['average_duration']:.1f} seconds")
        print(f"   Metadata saved: {metadata_path}")
        
        return metadata
    
    def create_training_structure(self, metadata: Dict):
        """Create F5-TTS compatible training structure"""
        print("\n🔧 Creating F5-TTS training structure...")
        
        # Create directories
        audio_dir = self.output_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        
        # Copy audio files with standardized names
        metadata_lines = []
        
        for i, sample in enumerate(metadata['samples']):
            # Copy audio file
            src_path = Path(sample['path'])
            dst_path = audio_dir / f"sample_{i:03d}.wav"
            
            # Convert to standard format (24kHz, mono)
            audio, sr = librosa.load(str(src_path), sr=24000, mono=True)
            sf.write(str(dst_path), audio, 24000)
            
            # Create metadata line for F5-TTS
            metadata_line = f"sample_{i:03d}.wav|{sample['transcription']}"
            metadata_lines.append(metadata_line)
            
            print(f"   📁 Copied: {src_path.name} → {dst_path.name}")
        
        # Save F5-TTS metadata file
        metadata_file = self.output_dir / "metadata.txt"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(metadata_lines))
        
        print(f"✅ F5-TTS structure created")
        print(f"   Audio files: {audio_dir}")
        print(f"   Metadata: {metadata_file}")
        
        return self.output_dir
    
    def create_finetune_config(self):
        """Create configuration for fine-tuning"""
        config = {
            "model_name": "F5TTS_Base",
            "data_path": str(self.output_dir),
            "batch_size": 4,
            "learning_rate": 1e-4,
            "num_epochs": 100,
            "save_every": 10,
            "sample_rate": 24000,
            "notes": "Fine-tuning with left_speaker voice samples"
        }
        
        config_path = self.output_dir / "finetune_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"⚙️  Fine-tune config: {config_path}")
        return config_path

def main():
    # Configure paths
    samples_dir = "/Users/adamleeperelman/Documents/LLM/Voice Clone/ai_voice_samples/left_speaker"
    output_dir = "/Users/adamleeperelman/Documents/LLM/Voice Clone/F5_TTS/finetune_data/left_speaker_training"
    
    print("🎯 F5-TTS Training Data Preparation")
    print(f"📁 Samples: {samples_dir}")
    print(f"💾 Output: {output_dir}")
    
    # Initialize trainer
    trainer = SimpleVoiceTrainer(samples_dir, output_dir)
    
    try:
        # Prepare training data
        metadata = trainer.prepare_training_data()
        
        if metadata['total_samples'] == 0:
            print("❌ No valid samples found")
            return 1
        
        # Create training structure
        training_dir = trainer.create_training_structure(metadata)
        
        # Create config
        config_path = trainer.create_finetune_config()
        
        print(f"\n🎉 Training preparation complete!")
        print(f"📁 Training directory: {training_dir}")
        print(f"⚙️  Config file: {config_path}")
        
        print(f"\n📚 Next steps:")
        print(f"1. Install F5-TTS: pip install f5-tts")
        print(f"2. Run training with your prepared data")
        print(f"3. Use the fine-tuned model for voice cloning")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
