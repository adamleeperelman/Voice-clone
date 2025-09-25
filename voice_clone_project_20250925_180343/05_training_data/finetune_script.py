
import os
import sys
sys.path.append('/Users/adamleeperelman/Documents/LLM/Voice Clone')

# Import F5-TTS fine-tuning modules
try:
    from f5_tts.train import train_model
    from f5_tts.configs import TrainingConfig
    
    # Configure training
    config = TrainingConfig(
        data_dir='/Users/adamleeperelman/Documents/LLM/Voice Clone/voice_clone_project_20250925_180343/05_training_data',
        model_name='main_speaker_model',
        epochs=100,
        learning_rate=0.0001,
        batch_size=4,
        save_dir='/Users/adamleeperelman/Documents/LLM/Voice Clone/voice_clone_project_20250925_180343/05_training_data/checkpoints'
    )
    
    # Start training
    print("🚀 Starting F5-TTS fine-tuning...")
    train_model(config)
    print("✅ Fine-tuning completed!")
    
except ImportError as e:
    print(f"❌ F5-TTS training modules not found: {e}")
    print("This is a template - actual F5-TTS fine-tuning may require different setup")
    
except Exception as e:
    print(f"❌ Fine-tuning error: {e}")
