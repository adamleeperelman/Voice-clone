#!/usr/bin/env python3
"""
Voice AI Separator - No naming conflicts
Uses OpenAI Whisper + GPT for intelligent speaker separation
"""

import os
import sys
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pathlib import Path
import json
from openai import OpenAI
import warnings
warnings.filterwarnings("ignore")

# Import whisper safely
def load_whisper():
    try:
        # Try different import methods
        import whisper
        return whisper
    except Exception as e:
        print(f"❌ Error loading Whisper: {e}")
        print("Install with: pip install openai-whisper")
        sys.exit(1)

class VoiceAISeparator:
    def __init__(self, openai_api_key=None):
        """Initialize with OpenAI API key"""
        self.client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        
        # Load Whisper model
        print("🤖 Loading Whisper model...")
        self.whisper = load_whisper()
        self.whisper_model = self.whisper.load_model("base")  # Start with base model
        print("✅ Whisper model loaded")
    
    def transcribe_audio(self, audio_file_path):
        """
        Transcribe audio with timestamps
        """
        print("🎤 Transcribing audio...")
        
        try:
            result = self.whisper_model.transcribe(
                audio_file_path,
                word_timestamps=True,
                language="en",
                # Better segmentation parameters
                temperature=0.0,  # More consistent results
                best_of=1,        # Faster processing
                beam_size=1       # More deterministic
            )
            return result
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            return None
    
    def merge_short_segments(self, segments, min_duration=3.0, max_duration=15.0):
        """
        Merge short segments to create more natural speech units
        """
        if not segments:
            return segments
            
        merged_segments = []
        current_segment = None
        
        for segment in segments:
            duration = segment['end'] - segment['start']
            
            if current_segment is None:
                current_segment = segment.copy()
            elif (current_segment['end'] - current_segment['start']) < min_duration:
                # Merge with current segment if it's too short
                current_segment['end'] = segment['end']
                current_segment['text'] += segment['text']
            elif (current_segment['end'] - current_segment['start']) > max_duration:
                # Current segment is too long, finalize it and start new one
                merged_segments.append(current_segment)
                current_segment = segment.copy()
            else:
                # Current segment is good size, finalize it and start new one
                merged_segments.append(current_segment)
                current_segment = segment.copy()
        
        # Don't forget the last segment
        if current_segment is not None:
            merged_segments.append(current_segment)
            
        return merged_segments

    def analyze_with_gpt(self, transcription_data):
        """
        Use GPT to find the best segments
        """
        if not self.client:
            print("⚠️  No OpenAI API - using basic analysis")
            return self.basic_analysis(transcription_data)
        
        print("🧠 Analyzing with GPT...")
        
        # First, merge short segments for better speech units
        merged_segments = self.merge_short_segments(transcription_data['segments'])
        print(f"📝 Merged {len(transcription_data['segments'])} segments into {len(merged_segments)} better units")
        
        # Prepare segments for analysis
        segments_text = []
        for segment in merged_segments:
            start = segment['start']
            end = segment['end']
            text = segment['text'].strip()
            duration = end - start
            segments_text.append(f"[{start:.1f}s-{end:.1f}s] ({duration:.1f}s): {text}")
        
        # Limit to first 30 segments for API efficiency
        analysis_text = "\n".join(segments_text[:30])
        
        prompt = f"""
Analyze this audio transcription to find the BEST segments for voice cloning training.

Requirements:
- Segments should be 8-15 seconds long
- Only ONE person speaking (no overlaps)
- Clear, high-quality speech
- Good content (complete sentences/phrases)

Transcription:
{analysis_text}

Return a JSON list of the 5-8 BEST segments:
[
  {{"start": 0.0, "end": 12.5, "reason": "clear single speaker, good length", "quality": 9}},
  {{"start": 25.3, "end": 37.1, "reason": "complete sentences, no background", "quality": 8}}
]

Focus on segments with:
1. Complete thoughts/sentences
2. Clear pronunciation
3. No interruptions or overlaps
4. Optimal 8-15 second duration
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in selecting optimal audio segments for voice cloning training. Focus on quality over quantity."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            # Extract JSON from response
            response_text = response.choices[0].message.content
            
            # Find JSON in response
            import re
            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if json_match:
                recommendations = json.loads(json_match.group())
                return recommendations
            else:
                print("⚠️  GPT response format issue, using fallback")
                return self.basic_analysis(transcription_data)
                
        except Exception as e:
            print(f"⚠️  GPT analysis failed: {e}")
            return self.basic_analysis(transcription_data)
    
    def basic_analysis(self, transcription_data):
        """
        Basic analysis without GPT
        """
        print("📝 Using basic analysis...")
        
        # First, merge short segments for better speech units
        merged_segments = self.merge_short_segments(transcription_data['segments'])
        print(f"📝 Merged {len(transcription_data['segments'])} segments into {len(merged_segments)} better units")
        
        recommendations = []
        
        for segment in merged_segments:
            duration = segment['end'] - segment['start']
            text_length = len(segment['text'].strip())
            
            # Select segments with good duration and content
            if 8 <= duration <= 15 and text_length > 20:
                quality = 7 if text_length > 50 else 6
                recommendations.append({
                    "start": segment['start'],
                    "end": segment['end'],
                    "reason": f"good duration ({duration:.1f}s) and content length",
                    "quality": quality
                })
        
        # Sort by quality and return top 8
        recommendations.sort(key=lambda x: x['quality'], reverse=True)
        return recommendations[:8]
    
    def extract_samples(self, audio_file_path, output_dir="ai_voice_samples"):
        """
        Main extraction function
        """
        Path(output_dir).mkdir(exist_ok=True)
        Path(f"{output_dir}/left_speaker").mkdir(exist_ok=True)
        Path(f"{output_dir}/right_speaker").mkdir(exist_ok=True)
        Path(f"{output_dir}/analysis").mkdir(exist_ok=True)
        
        print(f"📁 Loading: {audio_file_path}")
        
        # Load stereo audio
        stereo_audio = AudioSegment.from_file(audio_file_path)
        
        if stereo_audio.channels != 2:
            print(f"❌ Expected stereo, got {stereo_audio.channels} channels")
            return False
        
        print(f"✅ Loaded: {len(stereo_audio)/1000:.1f} seconds")
        
        # Split channels
        left_channel = stereo_audio.split_to_mono()[0]
        right_channel = stereo_audio.split_to_mono()[1]
        
        # Save temporary files for Whisper
        left_temp = f"{output_dir}/temp_left.wav"
        right_temp = f"{output_dir}/temp_right.wav"
        left_channel.export(left_temp, format="wav")
        right_channel.export(right_temp, format="wav")
        
        # Process left channel
        print("\n🎤 Processing LEFT channel...")
        left_transcription = self.transcribe_audio(left_temp)
        
        if left_transcription:
            left_recommendations = self.analyze_with_gpt(left_transcription)
            left_count = self.save_recommended_samples(
                left_channel, left_recommendations, f"{output_dir}/left_speaker", "LEFT"
            )
            
            # Save analysis
            with open(f"{output_dir}/analysis/left_analysis.json", "w") as f:
                json.dump({
                    "transcription": left_transcription,
                    "recommendations": left_recommendations
                }, f, indent=2)
        else:
            left_count = 0
        
        # Process right channel
        print("\n🎤 Processing RIGHT channel...")
        right_transcription = self.transcribe_audio(right_temp)
        
        if right_transcription:
            right_recommendations = self.analyze_with_gpt(right_transcription)
            right_count = self.save_recommended_samples(
                right_channel, right_recommendations, f"{output_dir}/right_speaker", "RIGHT"
            )
            
            # Save analysis
            with open(f"{output_dir}/analysis/right_analysis.json", "w") as f:
                json.dump({
                    "transcription": right_transcription,
                    "recommendations": right_recommendations
                }, f, indent=2)
        else:
            right_count = 0
        
        # Cleanup
        os.remove(left_temp)
        os.remove(right_temp)
        
        # Summary
        print("\n" + "="*60)
        print("✅ AI VOICE SEPARATION COMPLETE!")
        print("="*60)
        print(f"📁 Output: {output_dir}/")
        print(f"🎤 Left speaker: {left_count} samples")
        print(f"🎤 Right speaker: {right_count} samples")
        print(f"📊 Analysis: {output_dir}/analysis/")
        
        return True
    
    def save_recommended_samples(self, channel, recommendations, output_dir, channel_name):
        """
        Save the recommended samples
        """
        if not recommendations:
            print(f"    ❌ No recommendations for {channel_name}")
            return 0
        
        print(f"  Saving {len(recommendations)} recommended samples for {channel_name}...")
        
        saved_count = 0
        
        for i, rec in enumerate(recommendations):
            start_ms = int(rec["start"] * 1000)
            end_ms = int(rec["end"] * 1000)
            
            # Extract sample
            sample = channel[start_ms:end_ms]
            
            # Basic enhancement
            sample = sample.normalize()
            
            # Save
            duration = len(sample) / 1000
            quality = rec.get("quality", 7)
            reason = rec.get("reason", "AI recommended")[:30]  # Truncate for filename
            
            filename = f"{output_dir}/AI_{saved_count+1:03d}_{duration:.1f}s_q{quality}.wav"
            sample.export(filename, format="wav")
            saved_count += 1
            
            print(f"    ✅ {channel_name}: sample {saved_count:03d} - {duration:.1f}s (Q:{quality}) - {reason}")
        
        return saved_count

def main():
    """Main function"""
    print("🤖 AI VOICE SEPARATOR")
    print("="*50)
    
    # Get API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key as an environment variable:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        sys.exit(1)
    print("✅ OpenAI API configured")
    
    # Get audio file
    if len(sys.argv) > 1:
        audio_file = ' '.join(sys.argv[1:]).strip('"\'')
    else:
        audio_file = input("\nEnter path to your stereo audio file: ").strip('"\'')
    
    if not os.path.exists(audio_file):
        print(f"❌ File not found: {audio_file}")
        return
    
    # Create separator and process
    separator = VoiceAISeparator(openai_api_key=api_key)
    success = separator.extract_samples(audio_file)
    
    if success:
        print("\n🎉 SUCCESS! AI-selected voice samples ready for training!")

if __name__ == "__main__":
    main()