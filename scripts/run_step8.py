from pathlib import Path
import sys

from complete_voice_pipeline_new import VoicePipeline


def main():
    project_root = sys.argv[1] if len(sys.argv) > 1 else "voice_clone_project_20250926_113823"
    project_path = Path(project_root)
    pipeline = VoicePipeline(project_dir=project_path)
    pipeline.step2_initialize_processor()
    result = pipeline.step8_fine_tune_model(use_base_model=False, quick_test=True)
    print("Step 8 result:", result)


if __name__ == "__main__":
    main()
