#!/usr/bin/env python3
"""
Voice AI Separator
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
        import whisper
        return whisper
    except Exception as e:
        print(f"❌ Error loading Whisper: {e}")
        print("Install with: pip install openai-whisper")
        sys.exit(1)

class VoiceAISeparator:
    def __init__(self, openai_api_key=None):
        """Initialize the Voice AI Separator"""
        self.whisper_model = None
        self.openai_client = None
        
        # Load environment variables
        if not openai_api_key:
            openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)
            print("✅ OpenAI client initialized")
        else:
            print("⚠️  No OpenAI API key found - will use simpler separation")
        
        print("🤖 Voice AI Separator initialized")
    
    def load_whisper_model(self, model_size="base"):
        """Load Whisper model for transcription"""
        if self.whisper_model is None:
            print(f"🔥 Loading Whisper model ({model_size})...")
            whisper = load_whisper()
            self.whisper_model = whisper.load_model(model_size)
            print("✅ Whisper model loaded")
        return self.whisper_model
    
    def extract_segments(self, audio_path, min_silence_len=2000, silence_thresh=-35, 
                        min_segment_len=5000, max_segment_len=15000):
        """Extract audio segments with better parameters to reduce segment count"""
        print(f"🎵 Loading audio: {audio_path}")
        
        # Load audio
        audio = AudioSegment.from_file(audio_path)
        
        # Convert to mono for analysis, keep original for separation
        mono_audio = audio.set_channels(1)
        
        # Split on silence with more aggressive parameters
        segments = split_on_silence(
            mono_audio,
            min_silence_len=min_silence_len,    # Longer silence required (2s vs 1s)
            silence_thresh=silence_thresh,      # Less sensitive (-35 vs -40)
            keep_silence=1000                   # Keep more silence padding
        )
        
        print(f"📊 Initial segments found: {len(segments)}")
        
        # Merge short segments and split long ones
        merged_segments = self.optimize_segments(segments, min_segment_len, max_segment_len)
        
        print(f"📊 Optimized segments: {len(merged_segments)} (reduced from {len(segments)})")
        return merged_segments, audio
    
    def optimize_segments(self, segments, min_len=5000, max_len=15000):
        """Merge short segments and split long ones to get optimal segment lengths"""
        optimized = []
        current_segment = None
        
        for segment in segments:
            segment_len = len(segment)
            
            # If segment is too long, split it
            if segment_len > max_len:
                # If we have a current segment, save it first
                if current_segment:
                    optimized.append(current_segment)
                    current_segment = None
                
                # Split long segment into chunks
                chunk_size = max_len
                for i in range(0, segment_len, chunk_size):
                    chunk = segment[i:i + chunk_size]
                    if len(chunk) >= min_len:  # Only keep chunks that meet minimum
                        optimized.append(chunk)
                continue
            
            # If segment is too short, try to merge
            if segment_len < min_len:
                if current_segment is None:
                    current_segment = segment
                else:
                    # Check if merging would exceed max length
                    if len(current_segment) + segment_len <= max_len:
                        current_segment += segment  # Merge segments
                    else:
                        # Save current and start new
                        if len(current_segment) >= min_len:
                            optimized.append(current_segment)
                        current_segment = segment
            else:
                # Segment is good length
                if current_segment:
                    # Check if we should merge or save separately
                    if len(current_segment) + segment_len <= max_len:
                        optimized.append(current_segment + segment)
                        current_segment = None
                    else:
                        if len(current_segment) >= min_len:
                            optimized.append(current_segment)
                        optimized.append(segment)
                        current_segment = None
                else:
                    optimized.append(segment)
        
        # Don't forget the last segment
        if current_segment and len(current_segment) >= min_len:
            optimized.append(current_segment)
        
        return optimized
    
    def transcribe_segments(self, segments):
        """Transcribe segments using Whisper"""
        if not self.whisper_model:
            self.load_whisper_model()
        
        transcriptions = []
        print("🎤 Transcribing segments...")
        
        for i, segment in enumerate(segments):
            try:
                # Save segment temporarily
                temp_path = f"temp_segment_{i}.wav"
                segment.export(temp_path, format="wav")
                
                # Transcribe with detailed output
                result = self.whisper_model.transcribe(
                    temp_path,
                    language="en",           # Specify language
                    task="transcribe",       # Keep original language
                    temperature=0.0,         # More deterministic
                    verbose=False,           # Quiet output
                    word_timestamps=True,    # Get timing info
                    condition_on_previous_text=False,  # Independent segments
                    compression_ratio_threshold=2.4,   # Filter low quality
                    logprob_threshold=-1.0,           # Filter uncertain words
                    no_speech_threshold=0.6           # Detect silence
                )
                
                # Calculate quality metrics
                avg_logprob = result.get('avg_logprob', -1.0)
                no_speech_prob = result.get('no_speech_prob', 1.0)
                compression_ratio = len(result['text']) / len(result['text'].split()) if result['text'] else 0
                
                # Audio quality checks
                audio_array = np.array(segment.get_array_of_samples())
                if segment.channels == 2:
                    audio_array = audio_array.reshape((-1, 2))
                    audio_array = audio_array.mean(axis=1)  # Convert to mono
                
                # Calculate RMS (loudness) and dynamic range
                rms = np.sqrt(np.mean(audio_array**2))
                dynamic_range = np.max(np.abs(audio_array)) - np.min(np.abs(audio_array))
                
                transcriptions.append({
                    'index': i,
                    'text': result['text'].strip(),
                    'duration': len(segment) / 1000.0,
                    'avg_logprob': avg_logprob,
                    'no_speech_prob': no_speech_prob,
                    'compression_ratio': compression_ratio,
                    'rms': float(rms),
                    'dynamic_range': float(dynamic_range),
                    'word_count': len(result['text'].split()) if result['text'] else 0,
                    'segments': result.get('segments', [])
                })
                
                # Clean up
                os.remove(temp_path)
                
                if i % 10 == 0:
                    print(f"   Processed {i+1}/{len(segments)} segments")
                    
            except Exception as e:
                print(f"⚠️  Error transcribing segment {i}: {e}")
                transcriptions.append({
                    'index': i,
                    'text': '',
                    'duration': len(segment) / 1000.0,
                    'avg_logprob': -2.0,
                    'no_speech_prob': 1.0,
                    'compression_ratio': 0,
                    'rms': 0,
                    'dynamic_range': 0,
                    'word_count': 0,
                    'segments': []
                })
        
        print(f"✅ Transcribed {len(transcriptions)} segments")
        return transcriptions
    
    def filter_high_quality_segments(self, transcriptions, min_duration=8.0, min_confidence=-0.5, 
                                   max_no_speech=0.3, min_words=10, min_rms=0.01):
        """Filter segments for high quality voice samples"""
        print("🔍 Filtering for high-quality segments...")
        
        # First, let's analyze the data distribution
        durations = [t['duration'] for t in transcriptions if t['text']]
        confidences = [t['avg_logprob'] for t in transcriptions if t['text']]
        no_speech_probs = [t['no_speech_prob'] for t in transcriptions if t['text']]
        word_counts = [t['word_count'] for t in transcriptions if t['text']]
        
        if durations:
            print(f"📊 Data Analysis:")
            print(f"   Durations: min={min(durations):.1f}s, max={max(durations):.1f}s, avg={sum(durations)/len(durations):.1f}s")
            print(f"   Confidence: min={min(confidences):.2f}, max={max(confidences):.2f}, avg={sum(confidences)/len(confidences):.2f}")
            print(f"   No-speech prob: min={min(no_speech_probs):.2f}, max={max(no_speech_probs):.2f}, avg={sum(no_speech_probs)/len(no_speech_probs):.2f}")
            print(f"   Word counts: min={min(word_counts)}, max={max(word_counts)}, avg={sum(word_counts)/len(word_counts):.1f}")
            
            # Count segments by duration ranges
            duration_ranges = {
                '< 5s': len([d for d in durations if d < 5]),
                '5-8s': len([d for d in durations if 5 <= d < 8]),
                '8-12s': len([d for d in durations if 8 <= d < 12]),
                '12s+': len([d for d in durations if d >= 12])
            }
            print(f"   Duration distribution: {duration_ranges}")
        
        high_quality = []
        stats = {
            'total': len(transcriptions),
            'too_short': 0,
            'low_confidence': 0,
            'too_much_silence': 0,
            'too_few_words': 0,
            'low_audio_quality': 0,
            'passed': 0
        }
        
        # Adaptive thresholds based on data distribution
        if durations and len(durations) > 10:
            # Use more reasonable thresholds
            duration_75th = sorted(durations)[int(len(durations) * 0.75)]
            confidence_75th = sorted(confidences)[int(len(confidences) * 0.75)]
            
            # Adjust thresholds if they're too strict
            adaptive_min_duration = min(min_duration, max(5.0, duration_75th * 0.7))
            adaptive_min_confidence = min(min_confidence, confidence_75th - 0.3)
            adaptive_min_words = max(5, min_words // 2)  # Reduce word requirement
            
            print(f"🔧 Adaptive thresholds:")
            print(f"   Duration: {adaptive_min_duration:.1f}s (was {min_duration:.1f}s)")
            print(f"   Confidence: {adaptive_min_confidence:.2f} (was {min_confidence:.2f})")
            print(f"   Min words: {adaptive_min_words} (was {min_words})")
        else:
            adaptive_min_duration = min_duration
            adaptive_min_confidence = min_confidence
            adaptive_min_words = min_words
        
        for t in transcriptions:
            # Skip empty transcriptions
            if not t['text']:
                continue
                
            # Check duration (at least adaptive minimum)
            if t['duration'] < adaptive_min_duration:
                stats['too_short'] += 1
                continue
            
            # Check confidence (higher avg_logprob is better)
            if t['avg_logprob'] < adaptive_min_confidence:
                stats['low_confidence'] += 1
                continue
            
            # Check for silence (lower no_speech_prob is better, should be < 0.3)
            if t['no_speech_prob'] > max_no_speech:
                stats['too_much_silence'] += 1
                continue
            
            # Check word count (should have enough words)
            if t['word_count'] < adaptive_min_words:
                stats['too_few_words'] += 1
                continue
            
            # Check audio quality (RMS should indicate clear audio)
            if t['rms'] < min_rms:
                stats['low_audio_quality'] += 1
                continue
            
            # Passed all filters
            high_quality.append(t)
            stats['passed'] += 1
        
        # Print filtering stats
        print(f"📊 Quality filtering results:")
        print(f"   Total segments: {stats['total']}")
        print(f"   Too short (<{adaptive_min_duration:.1f}s): {stats['too_short']}")
        print(f"   Low confidence (<{adaptive_min_confidence:.2f}): {stats['low_confidence']}")
        print(f"   Too much silence (>{max_no_speech}): {stats['too_much_silence']}")
        print(f"   Too few words (<{adaptive_min_words}): {stats['too_few_words']}")
        print(f"   Low audio quality (<{min_rms}): {stats['low_audio_quality']}")
        print(f"   ✅ High quality segments: {stats['passed']}")
        
        return high_quality
    
    def analyze_speakers_with_gpt(self, high_quality_transcriptions):
        """Use GPT to analyze speaker patterns from high-quality segments only"""
        if not self.openai_client:
            print("⚠️  No OpenAI client - using simple separation")
            return self.simple_speaker_separation(high_quality_transcriptions)
        
        print("🧠 Analyzing speakers with GPT using high-quality segments...")
        
        # Prepare text for analysis with quality scores
        text_segments = []
        for t in high_quality_transcriptions:
            if t['text']:
                confidence_info = f"(conf: {t['avg_logprob']:.2f}, duration: {t['duration']:.1f}s)"
                text_segments.append(f"Segment {t['index']}: {t['text']} {confidence_info}")
        
        if not text_segments:
            print("❌ No high-quality transcriptions available for analysis")
            return {}, {}
        
        analysis_text = "\n".join(text_segments[:30])  # Use fewer but higher quality segments
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are analyzing audio segments to identify different speakers. Look for patterns in speech style, topics, formality, and speaker characteristics."
                    },
                    {
                        "role": "user",
                        "content": f"Analyze these transcribed audio segments and group them by speaker. Return a JSON with speaker assignments:\n\n{analysis_text}\n\nFormat: {{\"speaker_1\": [segment_indices], \"speaker_2\": [segment_indices]}}"
                    }
                ],
                temperature=0.3
            )
            
            result_text = response.choices[0].message.content
            
            # Try to extract JSON
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                speaker_assignments = json.loads(json_match.group())
                print(f"✅ GPT identified {len(speaker_assignments)} speakers")
                return speaker_assignments, high_quality_transcriptions
            else:
                print("⚠️  Could not parse GPT response - using simple separation")
                return self.simple_speaker_separation(high_quality_transcriptions)
                
        except Exception as e:
            print(f"⚠️  GPT analysis failed: {e} - using simple separation")
            return self.simple_speaker_separation(high_quality_transcriptions)
    
    def simple_speaker_separation(self, transcriptions):
        """Simple speaker separation based on alternating pattern"""
        print("🔄 Using simple alternating speaker separation")
        
        speaker_assignments = {"left_speaker": [], "right_speaker": []}
        
        for i, t in enumerate(transcriptions):
            if t['text']:  # Only assign non-empty transcriptions
                if i % 2 == 0:
                    speaker_assignments["left_speaker"].append(t['index'])
                else:
                    speaker_assignments["right_speaker"].append(t['index'])
        
        return speaker_assignments, transcriptions
    
    def save_separated_audio_filtered(self, original_audio, segments, speaker_assignments, 
                                    filtered_transcriptions, output_dir="ai_voice_samples"):
        """Save only high-quality separated audio files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create speaker directories
        for speaker in speaker_assignments:
            speaker_dir = os.path.join(output_dir, speaker)
            os.makedirs(speaker_dir, exist_ok=True)
        
        saved_files = []
        quality_stats = []
        
        # Create lookup for transcription data by index
        transcription_lookup = {t['index']: t for t in filtered_transcriptions}
        
        for speaker, segment_indices in speaker_assignments.items():
            if not segment_indices:
                continue
                
            speaker_dir = os.path.join(output_dir, speaker)
            
            for i, seg_idx in enumerate(segment_indices):
                if seg_idx < len(segments) and seg_idx in transcription_lookup:
                    segment = segments[seg_idx]
                    transcription = transcription_lookup[seg_idx]
                    duration = transcription['duration']
                    
                    # Enhanced filename with quality metrics
                    confidence = transcription['avg_logprob']
                    filename = f"AI_{i+1:03d}_{duration:.1f}s_conf{confidence:.2f}.wav"
                    filepath = os.path.join(speaker_dir, filename)
                    
                    # Export segment
                    segment.export(filepath, format="wav")
                    saved_files.append(filepath)
                    
                    # Track quality stats
                    quality_stats.append({
                        'file': filename,
                        'duration': duration,
                        'confidence': confidence,
                        'no_speech_prob': transcription['no_speech_prob'],
                        'word_count': transcription['word_count'],
                        'text': transcription['text'][:50] + "..." if len(transcription['text']) > 50 else transcription['text']
                    })
                    
                    print(f"💾 Saved high-quality: {filepath} (conf: {confidence:.2f})")
        
        # Save quality report
        quality_report_path = os.path.join(output_dir, "quality_report.json")
        with open(quality_report_path, 'w') as f:
            json.dump({
                'total_high_quality_files': len(saved_files),
                'average_confidence': sum(s['confidence'] for s in quality_stats) / len(quality_stats) if quality_stats else 0,
                'average_duration': sum(s['duration'] for s in quality_stats) / len(quality_stats) if quality_stats else 0,
                'files': quality_stats
            }, f, indent=2)
        
        print(f"✅ Saved {len(saved_files)} high-quality audio files to {output_dir}")
        print(f"📊 Quality report saved to: {quality_report_path}")
        return saved_files

    def save_separated_audio(self, original_audio, segments, speaker_assignments, output_dir="ai_voice_samples"):
        """Save separated audio files (legacy method)"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create speaker directories
        for speaker in speaker_assignments:
            speaker_dir = os.path.join(output_dir, speaker)
            os.makedirs(speaker_dir, exist_ok=True)
        
        saved_files = []
        
        for speaker, segment_indices in speaker_assignments.items():
            if not segment_indices:
                continue
                
            speaker_dir = os.path.join(output_dir, speaker)
            
            for i, seg_idx in enumerate(segment_indices):
                if seg_idx < len(segments):
                    segment = segments[seg_idx]
                    duration = len(segment) / 1000.0
                    
                    # Generate filename
                    filename = f"AI_{i+1:03d}_{duration:.1f}s_q7.wav"
                    filepath = os.path.join(speaker_dir, filename)
                    
                    # Export segment
                    segment.export(filepath, format="wav")
                    saved_files.append(filepath)
                    
                    print(f"💾 Saved: {filepath}")
        
        print(f"✅ Saved {len(saved_files)} audio files to {output_dir}")
        return saved_files
    
    def separate_audio(self, input_path, output_dir="ai_voice_samples", 
                      min_duration=8.0, min_confidence=-0.5,
                      silence_len=2000, silence_thresh=-35, 
                      min_segment_len=5.0, max_segment_len=15.0):
        """Main method to separate audio into speakers with quality filtering"""
        print(f"🚀 Starting voice separation: {input_path}")
        
        # Extract segments with custom parameters
        segments, original_audio = self.extract_segments(
            input_path,
            min_silence_len=silence_len,
            silence_thresh=silence_thresh,
            min_segment_len=int(min_segment_len * 1000),  # Convert to ms
            max_segment_len=int(max_segment_len * 1000)   # Convert to ms
        )
        
        if not segments:
            print("❌ No audio segments found")
            return []
        
        # Transcribe segments
        transcriptions = self.transcribe_segments(segments)
        
        # Filter for high quality segments
        high_quality_transcriptions = self.filter_high_quality_segments(
            transcriptions, 
            min_duration=min_duration,
            min_confidence=min_confidence
        )
        
        if not high_quality_transcriptions:
            print("❌ No high-quality segments found after filtering")
            return []
        
        # Analyze speakers using only high-quality segments
        speaker_assignments, filtered_transcriptions = self.analyze_speakers_with_gpt(high_quality_transcriptions)
        
        # Save separated audio using only high-quality segments
        saved_files = self.save_separated_audio_filtered(
            original_audio, segments, speaker_assignments, filtered_transcriptions, output_dir
        )
        
        # Save enhanced analysis with quality metrics
        analysis_dir = os.path.join(output_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Create lookup for high-quality transcriptions
        transcription_lookup = {t['index']: t for t in filtered_transcriptions}
        
        for speaker, indices in speaker_assignments.items():
            analysis_file = os.path.join(analysis_dir, f"{speaker}_analysis.json")
            with open(analysis_file, 'w') as f:
                speaker_transcriptions = []
                for i in indices:
                    if i in transcription_lookup:
                        speaker_transcriptions.append(transcription_lookup[i])
                
                json.dump({
                    'speaker': speaker,
                    'total_high_quality_segments': len(speaker_transcriptions),
                    'segment_indices': indices,
                    'average_confidence': sum(t['avg_logprob'] for t in speaker_transcriptions) / len(speaker_transcriptions) if speaker_transcriptions else 0,
                    'average_duration': sum(t['duration'] for t in speaker_transcriptions) / len(speaker_transcriptions) if speaker_transcriptions else 0,
                    'total_duration': sum(t['duration'] for t in speaker_transcriptions),
                    'transcriptions': speaker_transcriptions
                }, f, indent=2)
        
        print(f"🎉 Voice separation complete!")
        print(f"📁 Output: {output_dir}")
        return saved_files

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Separate audio by speakers using AI with quality filtering")
    parser.add_argument("input", help="Input audio file path")
    parser.add_argument("-o", "--output", default="ai_voice_samples", help="Output directory")
    parser.add_argument("--model", default="base", help="Whisper model size")
    parser.add_argument("--min-duration", type=float, default=8.0, 
                       help="Minimum duration for audio segments (seconds)")
    parser.add_argument("--min-confidence", type=float, default=-0.5,
                       help="Minimum confidence score (higher = more confident)")
    parser.add_argument("--silence-len", type=int, default=2000,
                       help="Minimum silence length in ms to split segments")
    parser.add_argument("--silence-thresh", type=int, default=-35,
                       help="Silence threshold in dB (lower = more sensitive)")
    parser.add_argument("--min-segment", type=float, default=5.0,
                       help="Minimum segment length in seconds")
    parser.add_argument("--max-segment", type=float, default=15.0,
                       help="Maximum segment length in seconds")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    # Initialize separator
    separator = VoiceAISeparator()
    separator.load_whisper_model(args.model)
    
    # Separate audio with quality filtering
    separator.separate_audio(
        args.input, 
        args.output,
        min_duration=args.min_duration,
        min_confidence=args.min_confidence,
        silence_len=args.silence_len,
        silence_thresh=args.silence_thresh,
        min_segment_len=args.min_segment,
        max_segment_len=args.max_segment
    )

if __name__ == "__main__":
    main()
