#!/usr/bin/env python3
"""
Example usage of the modular Voice AI system
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the Python path so we can import voice_ai
sys.path.append(str(Path(__file__).parent))

from voice_ai import VoiceAIProcessor, create_processor

def main():
    """Example demonstrating the modular Voice AI system"""
    
    print("🎯 Voice AI Modular System Example")
    print("=" * 50)
    
    # Initialize the processor
    processor = create_processor()
    
    # Get workspace status
    print("\n📊 Checking workspace status...")
    status = processor.get_workspace_status()
    
    print(f"Workspace: {status['workspace_path']}")
    print(f"Separated audio samples exist: {status['separation'].get('samples_directory_exists', False)}")
    print(f"Left speaker files: {status['separation'].get('left_speaker_files', 0)}")
    print(f"Right speaker files: {status['separation'].get('right_speaker_files', 0)}")
    
    # Check training data
    training_status = status['training']
    print(f"Training data prepared: {training_status.get('training_data_prepared', False)}")
    print(f"Training samples: {training_status.get('total_samples', 0)}")
    
    # Check synthesis capabilities
    synthesis_status = status['synthesis']
    print(f"F5-TTS available: {synthesis_status.get('f5_tts_available', False)}")
    print(f"Available models: {synthesis_status.get('available_models', [])}")
    print(f"Reference audio found: {synthesis_status.get('reference_audio_found', False)}")
    
    print("\n🎯 Available Methods:")
    print("=" * 50)
    
    print("Voice Separation:")
    print("  processor.separate_voices(input_path, output_dir)")
    
    print("\nTraining Data Preparation:")
    print("  processor.prepare_training_data(source_dir, output_dir, speaker_name)")
    print("  processor.validate_training_data(training_dir)")
    print("  processor.fine_tune_model(training_dir)")
    
    print("\nVoice Synthesis:")
    print("  processor.synthesize_speech(text, reference_audio, model)")
    print("  processor.batch_synthesize(texts, reference_audio)")
    print("  processor.test_voice_cloning(reference_audio)")
    print("  processor.synthesize_from_file(text_file, reference_audio)")
    
    print("\nFull Pipeline:")
    print("  processor.full_pipeline(input_audio, test_text, speaker_name)")
    
    # Example of individual component usage
    print("\n🔧 Individual Component Usage:")
    print("=" * 50)
    
    print("# Direct access to components:")
    print("separator = processor.separator")
    print("trainer = processor.trainer") 
    print("synthesizer = processor.synthesizer")
    
    print("\n# Or import components directly:")
    print("from voice_ai import VoiceAISeparator, VoiceTrainer, VoiceSynthesizer")
    
    # If we have samples, demonstrate a simple test
    if status['separation'].get('left_speaker_files', 0) > 0:
        print("\n🧪 Running a quick test...")
        try:
            test_files = processor.test_voice_cloning()
            if test_files:
                print(f"✅ Generated {len(test_files)} test audio files")
                for file in test_files:
                    print(f"   📄 {Path(file).name}")
            else:
                print("⚠️  Test voice cloning failed")
        except Exception as e:
            print(f"⚠️  Test error: {e}")
    
    print("\n🎉 Modular Voice AI system is ready!")
    print("Use the methods above to separate voices, prepare training data, and synthesize speech.")

if __name__ == "__main__":
    main()
