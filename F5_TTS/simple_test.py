#!/usr/bin/env python3
"""
F5-TTS Simple Test
Basic test without heavy dependencies
"""

import os
import sys
from pathlib import Path

def check_voice_samples():
    """Check if voice samples are available"""
    samples_dir = Path("../ai_voice_samples")
    
    print("🔍 Checking for voice samples...")
    print(f"📁 Looking in: {samples_dir.absolute()}")
    
    if not samples_dir.exists():
        print("❌ ai_voice_samples directory not found")
        print("💡 Run voice_ai_separator.py first to generate samples")
        return False
    
    # Check for speaker folders
    speaker_folders = [d for d in samples_dir.iterdir() 
                      if d.is_dir() and d.name.endswith('_speaker')]
    
    if not speaker_folders:
        print("❌ No speaker folders found")
        return False
    
    total_samples = 0
    for folder in speaker_folders:
        samples = list(folder.glob('*.wav'))
        total_samples += len(samples)
        print(f"  📂 {folder.name}: {len(samples)} samples")
        
        # Show first few samples
        for sample in samples[:3]:
            print(f"    🎵 {sample.name}")
    
    print(f"✅ Found {total_samples} total voice samples in {len(speaker_folders)} speakers")
    return True

def check_dependencies():
    """Check which dependencies are available"""
    print("\n🔍 Checking dependencies...")
    
    deps = [
        ("torch", "PyTorch"),
        ("torchaudio", "TorchAudio"), 
        ("numpy", "NumPy"),
        ("librosa", "Librosa"),
        ("soundfile", "SoundFile"),
        ("pydub", "PyDub"),
    ]
    
    available = []
    missing = []
    
    for module, name in deps:
        try:
            __import__(module)
            print(f"  ✅ {name}")
            available.append(name)
        except ImportError:
            print(f"  ❌ {name}")
            missing.append(name)
    
    # Check F5-TTS
    try:
        import f5_tts
        print(f"  ✅ F5-TTS")
        available.append("F5-TTS")
    except ImportError:
        print(f"  ❌ F5-TTS")
        missing.append("F5-TTS")
    
    print(f"\n📊 Summary: {len(available)} available, {len(missing)} missing")
    
    if missing:
        print("\n💡 To install missing dependencies:")
        print("   python setup.py")
        print("   # OR manually:")
        if "PyTorch" in missing:
            print("   pip install torch torchaudio")
        if "NumPy" in missing:
            print("   pip install numpy")
        if "Librosa" in missing:
            print("   pip install librosa")
        if "SoundFile" in missing:
            print("   pip install soundfile")
        if "PyDub" in missing:
            print("   pip install pydub")
        if "F5-TTS" in missing:
            print("   pip install git+https://github.com/SWivid/F5-TTS.git")
    
    return len(missing) == 0

def create_test_structure():
    """Create a basic test structure"""
    print("\n🔧 Setting up test environment...")
    
    # Create output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    print(f"  📁 Created: {output_dir}")
    
    # Create a simple test script
    test_script = '''#!/usr/bin/env python3
"""Generated test script"""

print("🧪 F5-TTS Test")
print("This is a placeholder test.")
print("Install dependencies and run example.py for full functionality.")
'''
    
    test_file = output_dir / "simple_test.py"
    with open(test_file, 'w') as f:
        f.write(test_script)
    
    print(f"  📝 Created: {test_file}")
    
    return True

def main():
    """Main test function"""
    print("🧪 F5-TTS Simple Test")
    print("="*40)
    
    # Check voice samples
    samples_ok = check_voice_samples()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Create test structure
    create_test_structure()
    
    print("\n📋 Test Results:")
    print(f"  Voice samples: {'✅' if samples_ok else '❌'}")
    print(f"  Dependencies: {'✅' if deps_ok else '❌'}")
    
    if samples_ok and deps_ok:
        print("\n🎉 Ready for voice cloning!")
        print("💡 Run: python example.py")
    elif samples_ok:
        print("\n⚠️  Voice samples ready, but dependencies missing")
        print("💡 Run: python setup.py")
    else:
        print("\n❌ Voice samples not found")
        print("💡 Run voice_ai_separator.py first")

if __name__ == "__main__":
    main()
