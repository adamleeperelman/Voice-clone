#!/usr/bin/env python3
"""
Voice AI Separator Module
Modular version of the voice separation functionality
"""

import os
import sys
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pathlib import Path
import json
from openai import OpenAI
from typing import Dict, List, Optional, Tuple
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
    """
    Modular Voice AI Separator
    Separates audio files by speakers using Whisper + GPT
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
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
    
    def load_whisper_model(self, model_size: str = "base") -> object:
        """Load Whisper model for transcription"""
        if self.whisper_model is None:
            print(f"🔥 Loading Whisper model ({model_size})...")
            whisper = load_whisper()
            self.whisper_model = whisper.load_model(model_size)
            print("✅ Whisper model loaded")
        return self.whisper_model
    
    def extract_time_range(self, 
                          input_path: str, 
                          start_minutes: float, 
                          end_minutes: float, 
                          output_path: str = None) -> str:
        """
        Extract a specific time range from an audio file
        
        Args:
            input_path: Path to the input audio file
            start_minutes: Start time in minutes (e.g., 0, 5.5, 10)
            end_minutes: End time in minutes (e.g., 15, 30.5, 60)
            output_path: Optional output path, auto-generated if not provided
        
        Returns:
            Path to the extracted audio file
        """
        print(f"✂️ Extracting audio range: {start_minutes:.1f} to {end_minutes:.1f} minutes")
        
        # Validate inputs
        if start_minutes < 0:
            raise ValueError("Start time cannot be negative")
        if end_minutes <= start_minutes:
            raise ValueError("End time must be greater than start time")
        
        # Load the audio file
        print(f"🎵 Loading audio: {input_path}")
        try:
            audio = AudioSegment.from_file(input_path)
        except Exception as e:
            raise ValueError(f"Failed to load audio file: {e}")
        
        # Get audio duration in minutes
        duration_minutes = len(audio) / (1000 * 60)
        print(f"📊 Original audio duration: {duration_minutes:.1f} minutes")
        
        # Validate time range against audio duration
        if start_minutes >= duration_minutes:
            raise ValueError(f"Start time ({start_minutes:.1f}m) exceeds audio duration ({duration_minutes:.1f}m)")
        
        # Adjust end time if it exceeds audio duration
        if end_minutes > duration_minutes:
            print(f"⚠️  End time ({end_minutes:.1f}m) exceeds audio duration, adjusting to {duration_minutes:.1f}m")
            end_minutes = duration_minutes
        
        # Convert minutes to milliseconds
        start_ms = int(start_minutes * 60 * 1000)
        end_ms = int(end_minutes * 60 * 1000)
        
        # Extract the segment
        print(f"✂️ Extracting segment from {start_ms/1000:.1f}s to {end_ms/1000:.1f}s...")
        extracted_audio = audio[start_ms:end_ms]
        
        # Generate output path if not provided
        if not output_path:
            input_file = Path(input_path)
            filename = f"{input_file.stem}_{start_minutes:.1f}m_to_{end_minutes:.1f}m{input_file.suffix}"
            output_path = str(input_file.parent / filename)
        
        # Save the extracted audio
        print(f"💾 Saving extracted audio: {output_path}")
        extracted_audio.export(output_path, format="mp3")
        
        # Report results
        extracted_duration = len(extracted_audio) / (1000 * 60)
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        
        print(f"✅ Audio extraction complete!")
        print(f"   📁 Output: {output_path}")
        print(f"   ⏱️  Duration: {extracted_duration:.1f} minutes")
        print(f"   📊 File size: {file_size:.1f} MB")
        
        return output_path
    
    def separate_channels(self, 
                         input_path: str, 
                         output_dir: str = None) -> Dict[str, str]:
        """
        Separate stereo audio into left and right channel files
        
        Args:
            input_path: Path to the stereo input audio file
            output_dir: Directory to save channel files (auto-generated if not provided)
        
        Returns:
            Dictionary with 'left' and 'right' keys containing file paths
        """
        print(f"🔀 Separating audio channels: {input_path}")
        
        # Load the audio file
        try:
            audio = AudioSegment.from_file(input_path)
        except Exception as e:
            raise ValueError(f"Failed to load audio file: {e}")
        
        # Check if audio is stereo
        if audio.channels == 1:
            print("⚠️  Audio is mono - creating duplicate channels")
            left_audio = right_audio = audio
        elif audio.channels == 2:
            print("✅ Stereo audio detected - separating channels")
            # Split stereo into left and right channels
            left_audio = audio.split_to_mono()[0]  # Left channel
            right_audio = audio.split_to_mono()[1]  # Right channel
        else:
            print(f"⚠️  Audio has {audio.channels} channels - using first two")
            channels = audio.split_to_mono()
            left_audio = channels[0]
            right_audio = channels[1] if len(channels) > 1 else channels[0]
        
        # Generate output directory if not provided
        if not output_dir:
            input_file = Path(input_path)
            output_dir = input_file.parent / f"{input_file.stem}_channels"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate output file paths
        input_name = Path(input_path).stem
        left_path = output_path / f"{input_name}_left_channel.wav"
        right_path = output_path / f"{input_name}_right_channel.wav"
        
        # Export channels
        print(f"💾 Saving left channel: {left_path}")
        left_audio.export(str(left_path), format="wav")
        
        print(f"💾 Saving right channel: {right_path}")
        right_audio.export(str(right_path), format="wav")
        
        # Report results
        left_duration = len(left_audio) / (1000 * 60)
        right_duration = len(right_audio) / (1000 * 60)
        left_size = os.path.getsize(left_path) / (1024 * 1024)  # MB
        right_size = os.path.getsize(right_path) / (1024 * 1024)  # MB
        
        print(f"✅ Channel separation complete!")
        print(f"   📁 Output directory: {output_path}")
        print(f"   🎵 Left channel: {left_duration:.1f} min, {left_size:.1f} MB")
        print(f"   🎵 Right channel: {right_duration:.1f} min, {right_size:.1f} MB")
        
        return {
            'left': str(left_path),
            'right': str(right_path),
            'output_dir': str(output_path)
        }
    
    def detect_voice_activity(self, 
                             input_path: str,
                             min_voice_threshold: float = 0.02,
                             min_voice_ratio: float = 0.3,
                             frame_duration_ms: int = 30) -> Dict:
        """
        Detect voice activity in audio and filter out low voice or empty segments
        
        Args:
            input_path: Path to the audio file to analyze
            min_voice_threshold: Minimum RMS energy to consider as voice (0.01-0.1)
            min_voice_ratio: Minimum ratio of voice frames to total frames (0.1-0.8)
            frame_duration_ms: Duration of each analysis frame in milliseconds
        
        Returns:
            Dictionary with voice activity analysis results
        """
        print(f"🎤 Analyzing voice activity: {input_path}")
        
        try:
            # Load audio
            audio = AudioSegment.from_file(input_path)
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Calculate frame size
            frame_size = int(frame_duration_ms * audio.frame_rate / 1000)
            total_frames = len(audio.raw_data) // (frame_size * audio.sample_width)
            
            # Analyze audio in frames
            voice_frames = 0
            total_energy = 0
            max_energy = 0
            
            for i in range(total_frames):
                start_ms = i * frame_duration_ms
                end_ms = min((i + 1) * frame_duration_ms, len(audio))
                
                if start_ms >= len(audio):
                    break
                
                # Extract frame
                frame = audio[start_ms:end_ms]
                
                # Calculate RMS energy
                if len(frame.raw_data) > 0:
                    # Convert to numpy array for RMS calculation
                    samples = np.array(frame.get_array_of_samples())
                    rms_energy = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
                    
                    # Normalize by max possible value
                    normalized_energy = rms_energy / (2 ** (audio.sample_width * 8 - 1))
                    
                    total_energy += normalized_energy
                    max_energy = max(max_energy, normalized_energy)
                    
                    # Check if frame contains voice
                    if normalized_energy > min_voice_threshold:
                        voice_frames += 1
            
            # Calculate metrics
            voice_ratio = voice_frames / total_frames if total_frames > 0 else 0
            avg_energy = total_energy / total_frames if total_frames > 0 else 0
            duration_seconds = len(audio) / 1000
            
            # Determine if audio has sufficient voice activity
            has_voice = (voice_ratio >= min_voice_ratio and 
                        avg_energy > min_voice_threshold * 0.5)
            
            result = {
                'has_voice': has_voice,
                'voice_ratio': voice_ratio,
                'avg_energy': avg_energy,
                'max_energy': max_energy,
                'voice_frames': voice_frames,
                'total_frames': total_frames,
                'duration_seconds': duration_seconds,
                'analysis': {
                    'voice_threshold': min_voice_threshold,
                    'voice_ratio_threshold': min_voice_ratio,
                    'passed_voice_ratio': voice_ratio >= min_voice_ratio,
                    'passed_energy_check': avg_energy > min_voice_threshold * 0.5
                }
            }
            
            # Log results
            print(f"   📊 Voice activity analysis:")
            print(f"   ⏱️  Duration: {duration_seconds:.1f}s")
            print(f"   🎤 Voice ratio: {voice_ratio:.2f} (need ≥{min_voice_ratio:.2f})")
            print(f"   ⚡ Avg energy: {avg_energy:.4f} (need ≥{min_voice_threshold * 0.5:.4f})")
            print(f"   📈 Max energy: {max_energy:.4f}")
            print(f"   {'✅' if has_voice else '❌'} Voice detected: {has_voice}")
            
            return result
            
        except Exception as e:
            print(f"❌ Voice activity detection failed: {e}")
            return {
                'has_voice': False,
                'error': str(e),
                'voice_ratio': 0,
                'avg_energy': 0
            }
    
    def filter_audio_by_voice_activity(self, 
                                      input_paths: List[str],
                                      output_dir: str = None,
                                      min_voice_threshold: float = 0.02,
                                      min_voice_ratio: float = 0.3) -> List[str]:
        """
        Filter a list of audio files to keep only those with sufficient voice activity
        
        Args:
            input_paths: List of audio file paths to filter
            output_dir: Directory to copy filtered files (optional)
            min_voice_threshold: Minimum RMS energy threshold
            min_voice_ratio: Minimum ratio of voice frames
        
        Returns:
            List of paths to files with sufficient voice activity
        """
        print(f"🔍 Filtering {len(input_paths)} audio files by voice activity")
        
        voice_active_files = []
        filtered_out = []
        
        for i, file_path in enumerate(input_paths, 1):
            print(f"\n📂 Analyzing file {i}/{len(input_paths)}: {Path(file_path).name}")
            
            # Analyze voice activity
            analysis = self.detect_voice_activity(
                file_path, 
                min_voice_threshold, 
                min_voice_ratio
            )
            
            if analysis.get('has_voice', False):
                voice_active_files.append(file_path)
                
                # Copy to output directory if specified
                if output_dir:
                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)
                    
                    # Copy file to filtered directory
                    import shutil
                    dest_path = output_path / Path(file_path).name
                    shutil.copy2(file_path, dest_path)
                    print(f"   📋 Copied to: {dest_path}")
            else:
                filtered_out.append(file_path)
                print(f"   🗑️  Filtered out: insufficient voice activity")
        
        # Summary
        print(f"\n📊 Voice Activity Filtering Results:")
        print(f"   ✅ Files with voice: {len(voice_active_files)}")
        print(f"   ❌ Files filtered out: {len(filtered_out)}")
        print(f"   📈 Keep ratio: {len(voice_active_files)/len(input_paths)*100:.1f}%")
        
        if filtered_out:
            print(f"\n🗑️  Filtered out files:")
            for file_path in filtered_out:
                print(f"   - {Path(file_path).name}")
        
        return voice_active_files
    
    def extract_segments(self, 
                        audio_path: str, 
                        min_silence_len: int = 2000, 
                        silence_thresh: int = -35,
                        min_segment_len: int = 5000, 
                        max_segment_len: int = 15000) -> Tuple[List, AudioSegment]:
        """Extract audio segments with optimized parameters"""
        print(f"🎵 Loading audio: {audio_path}")
        
        # Load audio
        audio = AudioSegment.from_file(audio_path)
        
        # Convert to mono for analysis, keep original for separation
        mono_audio = audio.set_channels(1)
        
        # Split on silence with aggressive parameters
        segments = split_on_silence(
            mono_audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=1000
        )
        
        print(f"📊 Initial segments found: {len(segments)}")
        
        # Optimize segments
        optimized_segments = self._optimize_segments(segments, min_segment_len, max_segment_len)
        
        print(f"📊 Optimized segments: {len(optimized_segments)} (reduced from {len(segments)})")
        return optimized_segments, audio
    
    def _optimize_segments(self, segments: List, min_len: int = 5000, max_len: int = 15000) -> List:
        """Merge short segments and split long ones"""
        optimized = []
        current_segment = None
        
        for segment in segments:
            segment_len = len(segment)
            
            # If segment is too long, split it
            if segment_len > max_len:
                if current_segment:
                    optimized.append(current_segment)
                    current_segment = None
                
                # Split long segment into chunks
                chunk_size = max_len
                for i in range(0, segment_len, chunk_size):
                    chunk = segment[i:i + chunk_size]
                    if len(chunk) >= min_len:
                        optimized.append(chunk)
                continue
            
            # If segment is too short, try to merge
            if segment_len < min_len:
                if current_segment is None:
                    current_segment = segment
                else:
                    if len(current_segment) + segment_len <= max_len:
                        current_segment += segment
                    else:
                        if len(current_segment) >= min_len:
                            optimized.append(current_segment)
                        current_segment = segment
            else:
                # Segment is good length
                if current_segment:
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
    
    def transcribe_segments(self, segments: List) -> List[Dict]:
        """Transcribe segments using Whisper with quality metrics"""
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
                    language="en",
                    task="transcribe",
                    temperature=0.0,
                    verbose=False,
                    word_timestamps=True,
                    condition_on_previous_text=False,
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.6
                )
                
                # Calculate quality metrics
                avg_logprob = result.get('avg_logprob', -1.0)
                no_speech_prob = result.get('no_speech_prob', 1.0)
                
                # Audio quality checks
                audio_array = np.array(segment.get_array_of_samples())
                if segment.channels == 2:
                    audio_array = audio_array.reshape((-1, 2))
                    audio_array = audio_array.mean(axis=1)
                
                rms = np.sqrt(np.mean(audio_array**2))
                dynamic_range = np.max(np.abs(audio_array)) - np.min(np.abs(audio_array))
                
                transcriptions.append({
                    'index': i,
                    'text': result['text'].strip(),
                    'duration': len(segment) / 1000.0,
                    'avg_logprob': avg_logprob,
                    'no_speech_prob': no_speech_prob,
                    'compression_ratio': len(result['text']) / len(result['text'].split()) if result['text'] else 0,
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
    
    def filter_high_quality_segments(self, 
                                   transcriptions: List[Dict], 
                                   min_duration: float = 8.0, 
                                   min_confidence: float = -0.5,
                                   max_no_speech: float = 0.3, 
                                   min_words: int = 10, 
                                   min_rms: float = 0.01) -> List[Dict]:
        """Filter segments for high quality with adaptive thresholds"""
        print("🔍 Filtering for high-quality segments...")
        
        # Analyze data distribution
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
            
            # Adaptive thresholds
            duration_75th = sorted(durations)[int(len(durations) * 0.75)]
            confidence_75th = sorted(confidences)[int(len(confidences) * 0.75)]
            
            adaptive_min_duration = min(min_duration, max(5.0, duration_75th * 0.7))
            adaptive_min_confidence = min(min_confidence, confidence_75th - 0.3)
            adaptive_min_words = max(5, min_words // 2)
            
            print(f"🔧 Adaptive thresholds:")
            print(f"   Duration: {adaptive_min_duration:.1f}s (was {min_duration:.1f}s)")
            print(f"   Confidence: {adaptive_min_confidence:.2f} (was {min_confidence:.2f})")
            print(f"   Min words: {adaptive_min_words} (was {min_words})")
        else:
            adaptive_min_duration = min_duration
            adaptive_min_confidence = min_confidence
            adaptive_min_words = min_words
        
        # Filter segments
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
        
        for t in transcriptions:
            if not t['text']:
                continue
                
            if t['duration'] < adaptive_min_duration:
                stats['too_short'] += 1
                continue
            
            if t['avg_logprob'] < adaptive_min_confidence:
                stats['low_confidence'] += 1
                continue
            
            if t['no_speech_prob'] > max_no_speech:
                stats['too_much_silence'] += 1
                continue
            
            if t['word_count'] < adaptive_min_words:
                stats['too_few_words'] += 1
                continue
            
            if t['rms'] < min_rms:
                stats['low_audio_quality'] += 1
                continue
            
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
    
    def analyze_speakers_with_gpt(self, high_quality_transcriptions: List[Dict]) -> Tuple[Dict, List[Dict]]:
        """Use GPT to analyze speaker patterns from high-quality segments"""
        if not self.openai_client:
            print("⚠️  No OpenAI client - using simple separation")
            return self._simple_speaker_separation(high_quality_transcriptions)
        
        print("🧠 Analyzing speakers with GPT using high-quality segments...")
        
        # Prepare text for analysis
        text_segments = []
        for t in high_quality_transcriptions:
            if t['text']:
                confidence_info = f"(conf: {t['avg_logprob']:.2f}, duration: {t['duration']:.1f}s)"
                text_segments.append(f"Segment {t['index']}: {t['text']} {confidence_info}")
        
        if not text_segments:
            print("❌ No high-quality transcriptions available for analysis")
            return {}, {}
        
        analysis_text = "\n".join(text_segments[:30])
        
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
                return self._simple_speaker_separation(high_quality_transcriptions)
                
        except Exception as e:
            print(f"⚠️  GPT analysis failed: {e} - using simple separation")
            return self._simple_speaker_separation(high_quality_transcriptions)
    
    def _simple_speaker_separation(self, transcriptions: List[Dict]) -> Tuple[Dict, List[Dict]]:
        """Simple speaker separation based on alternating pattern"""
        print("🔄 Using simple alternating speaker separation")
        
        speaker_assignments = {"left_speaker": [], "right_speaker": []}
        
        for i, t in enumerate(transcriptions):
            if t['text']:
                if i % 2 == 0:
                    speaker_assignments["left_speaker"].append(t['index'])
                else:
                    speaker_assignments["right_speaker"].append(t['index'])
        
        return speaker_assignments, transcriptions
    
    def save_separated_audio(self, 
                           original_audio: AudioSegment, 
                           segments: List, 
                           speaker_assignments: Dict, 
                           filtered_transcriptions: List[Dict], 
                           output_dir: str) -> List[str]:
        """Save high-quality separated audio files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create speaker directories
        for speaker in speaker_assignments:
            speaker_dir = os.path.join(output_dir, speaker)
            os.makedirs(speaker_dir, exist_ok=True)
        
        saved_files = []
        quality_stats = []
        
        # Create lookup for transcription data
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
                    confidence = transcription['avg_logprob']
                    
                    # Enhanced filename with quality metrics
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
    
    def separate_audio(self, 
                      input_path: str, 
                      output_dir: str = "ai_voice_samples",
                      min_duration: float = 8.0, 
                      min_confidence: float = -0.5,
                      silence_len: int = 2000, 
                      silence_thresh: int = -35, 
                      min_segment_len: float = 8.0, 
                      max_segment_len: float = 15.0,
                      max_no_speech_prob: float = 0.3) -> List[str]:
        """Main method to separate audio into speakers with quality filtering"""
        print(f"🚀 Starting voice separation: {input_path}")
        
        # Extract segments with custom parameters
        segments, original_audio = self.extract_segments(
            input_path,
            min_silence_len=silence_len,
            silence_thresh=silence_thresh,
            min_segment_len=int(min_segment_len * 1000),
            max_segment_len=int(max_segment_len * 1000)
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
        
        # Save separated audio
        saved_files = self.save_separated_audio(
            original_audio, segments, speaker_assignments, filtered_transcriptions, output_dir
        )
        
        # Save enhanced analysis
        analysis_dir = os.path.join(output_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
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
