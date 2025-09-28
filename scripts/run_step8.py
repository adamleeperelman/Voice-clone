from pathlib import Path

from complete_voice_pipeline_new import VoicePipeline


def main():
    project_path = Path("voice_clone_project_20250926_113823")
    pipeline = VoicePipeline(project_dir=project_path)
    pipeline.step2_initialize_processor()
    result = pipeline.step8_fine_tune_model(use_base_model=False, quick_test=True)
    print("Step 8 result:", result)


if __name__ == "__main__":
    main()
