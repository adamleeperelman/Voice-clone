#!/usr/bin/env python3
"""
Diagnose and Fix Gibberish Speech Issues
Analyzes training data quality and provides solutions
"""

from voice_ai import create_processor
from pathlib import Path
import json

def analyze_training_quality(project_dir: str):
    """Analyze training data to identify gibberish issues"""
    
    print("🔍 Analyzing Training Data Quality\n")
    print("=" * 80)
    
    # Load metadata
    metadata_path = Path(project_dir) / "05_training_data" / "metadata.json"
    if not metadata_path.exists():
        print("❌ No training metadata found")
        return
    
    with open(metadata_path, 'r') as f:
        data = json.load(f)
    
    # Analyze samples
    total_samples = data['total_samples']
    samples = data['samples']
    
    # Quality checks
    issues = []
    good_samples = []
    
    print(f"\n📊 Overall Stats:")
    print(f"   Total samples: {total_samples}")
    print(f"   Total duration: {data['total_duration']:.1f}s")
    print(f"   Average duration: {data['average_duration']:.1f}s")
    
    print(f"\n🔍 Analyzing Sample Quality:\n")
    
    for i, sample in enumerate(samples, 1):
        text = sample.get('text', '')
        duration = sample.get('duration', 0)
        
        # Quality checks
        has_issue = False
        sample_issues = []
        
        # Check 1: Text clarity
        unclear_words = ['um', 'uh', 'yeah', 'like', '...', 'yada', 'buddy']
        if any(word in text.lower() for word in unclear_words):
            sample_issues.append("Contains filler words")
            has_issue = True
        
        # Check 2: Conversational markers
        if '?' in text or '!' in text:
            sample_issues.append("Conversational (questions/exclamations)")
        
        # Check 3: Incomplete sentences
        if text.endswith('...') or len(text) < 20:
            sample_issues.append("Incomplete or very short text")
            has_issue = True
        
        # Check 4: Duration
        if duration < 3:
            sample_issues.append(f"Too short ({duration:.1f}s)")
            has_issue = True
        elif duration > 25:
            sample_issues.append(f"Too long ({duration:.1f}s)")
        
        if has_issue:
            issues.append({
                'index': i,
                'file': sample['audio_path'],
                'text': text[:100],
                'issues': sample_issues
            })
        else:
            good_samples.append(sample)
    
    # Report findings
    print(f"\n📈 Quality Analysis Results:")
    print(f"   ✅ Good quality samples: {len(good_samples)}/{total_samples}")
    print(f"   ⚠️  Samples with issues: {len(issues)}/{total_samples}")
    print(f"   📊 Quality ratio: {len(good_samples)/total_samples*100:.1f}%")
    
    if issues:
        print(f"\n⚠️  Problematic Samples (first 5):")
        for issue_data in issues[:5]:
            print(f"\n   Sample {issue_data['index']}:")
            print(f"   Text: {issue_data['text']}...")
            print(f"   Issues: {', '.join(issue_data['issues'])}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    print("=" * 80)
    
    if len(good_samples) < total_samples * 0.3:
        print("\n❌ CRITICAL: Less than 30% good quality samples!")
        print("   → The training data has too much conversational/unclear speech")
        print("   → This causes the model to generate gibberish")
        print("\n   Solutions:")
        print("   1. Use a different audio source with clearer speech")
        print("   2. Extract a different time range from your audio")
        print("   3. Use audiobook or podcast audio (clearer speech)")
        
    elif len(good_samples) < total_samples * 0.7:
        print("\n⚠️  WARNING: 30-70% good quality samples")
        print("   → Training data is mixed quality")
        print("\n   Solutions:")
        print("   1. Re-run pipeline with stricter voice filtering")
        print("   2. Manually select better audio segments")
        print("   3. Increase minimum segment duration to 8+ seconds")
        
    else:
        print("\n✅ GOOD: Training data quality is acceptable")
        print("\n   To improve gibberish issues:")
        print("   1. Use reference text with synthesis (already implemented)")
        print("   2. Increase CFG strength to 3.0-4.0 for clearer pronunciation")
        print("   3. Use only the best quality reference audio")
        print("   4. Try the base model instead of fine-tuned (if overfitted)")
    
    # Best samples recommendation
    if good_samples:
        print(f"\n✨ Best Reference Audio Samples:")
        for i, sample in enumerate(good_samples[:3], 1):
            duration = sample['duration']
            text = sample['text'][:80]
            print(f"\n   {i}. {sample['audio_path']}")
            print(f"      Duration: {duration:.1f}s")
            print(f"      Text: {text}...")
    
    return good_samples, issues

def test_solutions(project_dir: str, good_samples: list):
    """Test different solutions to fix gibberish"""
    
    if not good_samples:
        print("\n❌ No good samples found to test with")
        return
    
    print(f"\n\n🧪 Testing Solutions to Fix Gibberish")
    print("=" * 80)
    
    processor = create_processor(workspace_path=project_dir)
    
    # Get best reference audio
    best_sample = good_samples[0]
    ref_audio_relative = best_sample['audio_path'].replace('wavs/', '')
    ref_audio = f"{project_dir}/05_training_data/wavs/{ref_audio_relative}"
    ref_text = best_sample['text']
    
    test_text = "This is a clear and simple test sentence for voice synthesis."
    
    tests = [
        {
            "name": "Solution 1: Base Model (No Fine-tuning)",
            "params": {
                "checkpoint_path": "NONE",  # Force base model
                "nfe_step": 64,
                "cfg_strength": 3.0,
                "reference_text": ref_text
            }
        },
        {
            "name": "Solution 2: Fine-tuned with Strong Guidance",
            "params": {
                "nfe_step": 64,
                "cfg_strength": 4.0,  # Very strong guidance
                "reference_text": ref_text
            }
        },
        {
            "name": "Solution 3: High Quality + Reference Text",
            "params": {
                "nfe_step": 96,  # Very high quality
                "cfg_strength": 2.5,
                "reference_text": ref_text
            }
        }
    ]
    
    print(f"\nUsing best reference audio: {Path(ref_audio).name}")
    print(f"Reference text: {ref_text[:80]}...\n")
    
    for i, test in enumerate(tests, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/3: {test['name']}")
        print(f"{'='*80}")
        
        params = test['params'].copy()
        output_path = f"{project_dir}/07_generated_audio/gibberish_fix/test_{i}.wav"
        
        # Handle base model case
        if params.get('checkpoint_path') == "NONE":
            params['checkpoint_path'] = None
        
        try:
            result = processor.synthesize_speech(
                text=test_text,
                reference_audio=ref_audio,
                output_path=output_path,
                **params
            )
            
            if result:
                print(f"✅ Generated: {Path(result).name}")
            else:
                print(f"❌ Failed")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n\n📁 Test Results Location:")
    print(f"   {project_dir}/07_generated_audio/gibberish_fix/")
    print("\n🎧 Listen to test_1.wav, test_2.wav, test_3.wav to compare clarity!")

def main():
    project_dir = "voice_clone_project_20251005_203311"
    
    # Analyze quality
    good_samples, issues = analyze_training_quality(project_dir)
    
    # Test solutions
    if good_samples:
        test_solutions(project_dir, good_samples)

if __name__ == "__main__":
    main()
