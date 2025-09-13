# Voice AI Separator 🎤🤖

An intelligent AI-powered tool that separates stereo audio recordings into individual speaker samples, perfect for voice cloning training. Uses OpenAI Whisper for transcription and GPT-4 for intelligent segment analysis.

## Features

- ✨ **AI-Powered Analysis**: Uses GPT-4 to intelligently select the best audio segments
- 🎤 **Speaker Separation**: Automatically splits stereo recordings into left/right channels
- 📝 **Smart Segmentation**: Merges short fragments into natural speech units
- 🎯 **Quality Selection**: Chooses optimal 8-15 second samples for voice training
- 📊 **Detailed Analysis**: Provides transcriptions and quality reports

## Requirements

- Python 3.8+
- OpenAI API key
- Stereo audio file (.mp3, .wav, etc.)

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/adamleeperelman/Voice-clone.git
cd Voice-clone
```

2. **Install dependencies:**
```bash
pip install openai-whisper openai pydub numpy pathlib
```

3. **Set up your OpenAI API key:**
```bash
export OPENAI_API_KEY='your-openai-api-key-here'
```

Or create a `.env` file:
```bash
cp .env.example .env
# Edit .env and add your API key
```

## Usage

### Basic Usage

```bash
python voice_ai_separator.py "path/to/your/audio.mp3"
```

### Example with the provided recording

```bash
python voice_ai_separator.py "Raw Sample/recording.mp3"
```

### Interactive Mode

If you don't provide a file path, the script will prompt you:

```bash
python voice_ai_separator.py
# Enter path to your stereo audio file: Raw Sample/recording.mp3
```

## Expected Output

The script will create the following structure:

```
ai_voice_samples/
├── left_speaker/
│   ├── AI_001_8.3s_q7.wav
│   ├── AI_002_8.1s_q7.wav
│   └── ...
├── right_speaker/
│   ├── AI_001_11.9s_q7.wav
│   ├── AI_002_13.8s_q7.wav
│   └── ...
└── analysis/
    ├── left_analysis.json
    └── right_analysis.json
```

## Example Run

```bash
$ python voice_ai_separator.py "Raw Sample/recording.mp3"

🤖 AI VOICE SEPARATOR
==================================================
✅ OpenAI API configured
🤖 Loading Whisper model...
✅ Whisper model loaded
📁 Loading: Raw Sample/recording.mp3
✅ Loaded: 3512.4 seconds

🎤 Processing LEFT channel...
🎤 Transcribing audio...
🧠 Analyzing with GPT...
📝 Merged 637 segments into 397 better units
  Saving 8 recommended samples for LEFT...
    ✅ LEFT: sample 001 - 8.3s (Q:7) - good duration (8.3s) and conte
    ✅ LEFT: sample 002 - 8.1s (Q:7) - good duration (8.1s) and conte
    ...

🎤 Processing RIGHT channel...
🎤 Transcribing audio...
🧠 Analyzing with GPT...
📝 Merged 394 segments into 172 better units
  Saving 8 recommended samples for RIGHT...
    ✅ RIGHT: sample 001 - 11.9s (Q:7) - good duration (11.9s) and cont
    ✅ RIGHT: sample 002 - 13.8s (Q:7) - good duration (13.8s) and cont
    ...

============================================================
✅ AI VOICE SEPARATION COMPLETE!
============================================================
📁 Output: ai_voice_samples/
🎤 Left speaker: 8 samples
🎤 Right speaker: 8 samples
📊 Analysis: ai_voice_samples/analysis/

🎉 SUCCESS! AI-selected voice samples ready for training!
```

## How It Works

1. **Audio Loading**: Loads stereo audio and splits into left/right channels
2. **Transcription**: Uses Whisper to transcribe each channel with timestamps
3. **Segment Merging**: Intelligently merges short segments into natural speech units
4. **AI Analysis**: GPT-4 analyzes transcriptions to find optimal segments
5. **Sample Extraction**: Extracts and normalizes the best audio samples
6. **Quality Scoring**: Assigns quality scores based on duration and content

## Configuration

### Audio Requirements
- **Format**: Stereo audio (2 channels)
- **Duration**: Segments should be 8-15 seconds long
- **Quality**: Clear speech with minimal background noise

### API Limits
- Uses OpenAI GPT-4 for analysis (falls back to basic analysis if quota exceeded)
- Processes up to 30 segments per channel for efficiency

## Troubleshooting

### Common Issues

**API Key Not Set:**
```
❌ Error: OPENAI_API_KEY environment variable not set
```
Solution: Set your OpenAI API key as shown in the installation steps.

**Whisper Not Installed:**
```
❌ Error loading Whisper: No module named 'whisper'
```
Solution: Install Whisper with `pip install openai-whisper`

**File Not Found:**
```
❌ File not found: path/to/file.mp3
```
Solution: Check the file path and ensure the file exists.

**Non-Stereo Audio:**
```
❌ Expected stereo, got 1 channels
```
Solution: Use a stereo audio file with separate left/right channels.

## File Structure

```
Voice Clone/
├── voice_ai_separator.py      # Main voice separation script
├── F5_TTS/                    # Voice cloning module
│   ├── voice_cloner.py        # Main voice cloner
│   ├── model.py               # F5-TTS model wrapper
│   ├── example.py             # Usage examples
│   ├── setup.py               # Dependency installer
│   └── README.md              # F5-TTS documentation
├── pyproject.toml             # Project configuration
├── .gitignore                 # Git ignore rules
├── .env.example               # Environment template
├── README.md                  # This file
├── Raw Sample/
│   └── recording.mp3          # Example input file
└── ai_voice_samples/          # Generated output (after running)
    ├── left_speaker/          # Left channel samples
    ├── right_speaker/         # Right channel samples
    └── analysis/              # Transcription analysis
```

## Voice Cloning with F5-TTS

After generating voice samples, you can use them for voice cloning:

### 1. Set up F5-TTS
```bash
cd F5_TTS
python setup.py
```

### 2. Run voice cloning examples
```bash
python example.py
```

### 3. Use in your code
```python
from F5_TTS.voice_cloner import VoiceCloner

cloner = VoiceCloner()
cloner.set_voice("left")  # Use left speaker samples
audio = cloner.generate_speech(
    "Hello, this is a cloned voice!",
    output_path="cloned_speech.wav"
)
```

See `F5_TTS/README.md` for detailed voice cloning documentation.

## Advanced Usage

### Custom Output Directory
```bash
python voice_ai_separator.py "recording.mp3" --output custom_output/
```

### Environment Variables
```bash
# Set API key permanently (add to ~/.zshrc or ~/.bashrc)
echo 'export OPENAI_API_KEY="your-key-here"' >> ~/.zshrc
source ~/.zshrc
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source. See LICENSE file for details.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the example output
3. Open an issue on GitHub

---

🎉 **Ready to create amazing voice samples!** Run the script with your stereo recording and get AI-selected samples perfect for voice cloning training.
