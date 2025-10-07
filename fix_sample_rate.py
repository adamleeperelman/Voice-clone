#!/usr/bin/env python3
"""
Fix Sample Rate Mismatch - Resample Training Data to 24kHz
This fixes gibberish caused by sample rate mismatch
"""

import librosa
import soundfile as sf
from pathlib import Path
import shutil
import json

def check_sample_rates(training_dir: Path):
    """Check all training data sample rates"""
    wav_files = list(training_dir.glob('*.wav'))
    
    sample_rates = {}
    for wav_file in wav_files:
        info = sf.info(str(wav_file))
        sr = info.samplerate
        if sr not in sample_rates:
            sample_rates[sr] = []
        sample_rates[sr].append(wav_file)
    
    return sample_rates

def resample_training_data(project_dir: str, target_sr: int = 24000):
    """Resample all training data to target sample rate"""
    
    training_dir = Path(project_dir) / "05_training_data" / "wavs"
    
    if not training_dir.exists():
        print(f"❌ Training directory not found: {training_dir}")
        return
    
    print("🔧 Resampling Training Data to 24kHz")
    print("=" * 80)
    
    # Check current sample rates
    sample_rates = check_sample_rates(training_dir)
    
    print(f"\n📊 Current Sample Rates:")
    for sr, files in sample_rates.items():
        print(f"   {sr} Hz: {len(files)} files")
    
    if target_sr in sample_rates and len(sample_rates) == 1:
        print(f"\n✅ All files already at {target_sr} Hz - no resampling needed")
        return
    
    print(f"\n🎯 Target Sample Rate: {target_sr} Hz (F5-TTS requirement)")
    
    # Create backup
    backup_dir = Path(project_dir) / "05_training_data" / "wavs_backup_16khz"
    if not backup_dir.exists():
        print(f"\n💾 Creating backup at: wavs_backup_16khz/")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        for wav_file in training_dir.glob('*.wav'):
            shutil.copy2(wav_file, backup_dir / wav_file.name)
        print(f"   ✅ Backed up {len(list(backup_dir.glob('*.wav')))} files")
    
    # Resample files
    print(f"\n🔄 Resampling files to {target_sr} Hz...")
    
    resampled_count = 0
    for sr, files in sample_rates.items():
        if sr == target_sr:
            print(f"   ⏭️  Skipping {len(files)} files already at {target_sr} Hz")
            continue
        
        for wav_file in files:
            try:
                # Load audio at original sample rate
                audio, original_sr = librosa.load(str(wav_file), sr=sr)
                
                # Resample to target
                audio_resampled = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
                
                # Save resampled audio
                sf.write(str(wav_file), audio_resampled, target_sr)
                
                resampled_count += 1
                print(f"   ✅ Resampled: {wav_file.name} ({sr} Hz → {target_sr} Hz)")
                
            except Exception as e:
                print(f"   ❌ Error resampling {wav_file.name}: {e}")
    
    print(f"\n✅ Resampling Complete!")
    print(f"   📊 Resampled {resampled_count} files to {target_sr} Hz")
    print(f"   💾 Original files backed up in: wavs_backup_16khz/")
    
    # Verify
    print(f"\n🔍 Verifying sample rates...")
    new_sample_rates = check_sample_rates(training_dir)
    
    print(f"\n📈 New Sample Rates:")
    for sr, files in new_sample_rates.items():
        status = "✅" if sr == target_sr else "⚠️"
        print(f"   {status} {sr} Hz: {len(files)} files")
    
    if target_sr in new_sample_rates and len(new_sample_rates) == 1:
        print(f"\n🎉 SUCCESS: All training data now at {target_sr} Hz!")
        print(f"\n💡 Next Steps:")
        print(f"   1. Re-run the fine-tuning step (step 8)")
        print(f"   2. Or just use the base model with correct sample rate")
        print(f"   3. Gibberish should be significantly reduced!")
    
    return resampled_count

def main():
    project_dir = "voice_clone_project_20251005_203311"
    
    print("🎯 Sample Rate Fix for F5-TTS")
    print("=" * 80)
    print("\n⚠️  Issue: Training data at 16kHz, but F5-TTS expects 24kHz")
    print("   This causes gibberish/poor quality output")
    print("\n🔧 Solution: Resample all training data to 24kHz\n")
    
    input("Press Enter to continue with resampling... ")
    
    resample_training_data(project_dir, target_sr=24000)

if __name__ == "__main__":
    main()
