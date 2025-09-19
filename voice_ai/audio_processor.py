#!/usr/bin/env python3
"""
Audio Processor Module
Handles audio preprocessing tasks like time extraction, channel separation, and voice activity detection
"""

import os
import numpy as np
from pydub import AudioSegment
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings("ignore")


class AudioProcessor:
    """
    Audio preprocessing and analysis utilities
    Handles time extraction, channel separation, and voice activity detection
    """
    
    def __init__(self):
        """Initialize the Audio Processor"""
        print("🎵 Audio Processor initialized")
    
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
