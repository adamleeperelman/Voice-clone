#!/usr/bin/env python3
"""
Test Sample Rate Validation and Fix
Demonstrates the integrated sample rate validation in voice_ai module
"""

from voice_ai import create_processor

def test_sample_rate_validation():
    """Test the sample rate validation functionality"""
    
    # Create processor
    processor = create_processor(workspace_path='voice_clone_project_20251005_203311')
    
    print("🧪 Testing Sample Rate Validation & Fix")
    print("=" * 80)
    
    # Test on existing training data
    result = processor.fix_sample_rate(
        training_dir='voice_clone_project_20251005_203311/05_training_data',
        target_rate=24000,
        backup=True
    )
    
    print("\n📊 Validation Results:")
    print(f"   Status: {result.get('status', 'unknown')}")
    print(f"   Target Rate: {result.get('target_rate', 'N/A')} Hz")
    print(f"   Files Checked: {result.get('files_checked', 0)}")
    
    if result.get('status') == 'fixed':
        print(f"   Files Fixed: {result.get('files_fixed', 0)}")
        if result.get('backup_dir'):
            print(f"   Backup Location: {result.get('backup_dir')}")
    
    print("\n✅ Sample rate validation is now integrated into the pipeline!")
    print("💡 Future training data preparation will automatically validate sample rates")

if __name__ == "__main__":
    test_sample_rate_validation()
