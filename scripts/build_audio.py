#!/usr/bin/env python3
"""
Audio building script to merge translated audio segments into final audio file.
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
from pathlib import Path


def load_transcription(transcription_path):
    """Load translated transcription from JSON file."""
    if not os.path.exists(transcription_path):
        print(f"Error: Transcription file not found: {transcription_path}")
        return None
    
    with open(transcription_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_silence(duration_seconds, output_path, sample_rate=22050):
    """Create a silent audio file of specified duration."""
    cmd = [
        "ffmpeg",
        "-f", "lavfi",
        "-i", f"anullsrc=channel_layout=mono:sample_rate={sample_rate}",
        "-t", str(duration_seconds),
        "-y",
        output_path
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating silence: {e}")
        return False


def build_merged_audio(transcription, audio_dir, output_path):
    """Build merged audio from translated segments with proper timing gaps."""
    
    print(f"Building merged audio from: {audio_dir}")
    print(f"Output path: {output_path}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sort segments by start time
    segments = sorted(transcription['segments'], key=lambda x: x['start_time'])
    
    print(f"Processing {len(segments)} segments with realistic timing...")
    
    # Create a temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_files = []
        
        # Add initial 0.5 second silence
        initial_silence_path = os.path.join(temp_dir, "silence_initial.wav")
        if create_silence(0.5, initial_silence_path):
            temp_files.append(initial_silence_path)
            print(f"  Added initial 0.5s silence")
        
        previous_end_time = None
        
        for i, segment in enumerate(segments):
            segment_id = segment['segment_id']
            speaker = segment['speaker']
            start_time = segment['start_time']
            end_time = segment['end_time']
            start_time_str = segment['start_time_str']
            end_time_str = segment['end_time_str']
            duration_str = segment['duration_str']
            
            # Calculate gap from previous segment
            if previous_end_time is not None:
                gap_duration = start_time - previous_end_time
                
                if gap_duration > 0.1:  # Only add silence if gap is > 0.1 seconds
                    gap_silence_path = os.path.join(temp_dir, f"silence_gap_{i:03d}.wav")
                    if create_silence(gap_duration, gap_silence_path):
                        temp_files.append(gap_silence_path)
                        print(f"  Added {gap_duration:.3f}s gap before {segment_id}")
                elif gap_duration < 0:
                    print(f"  Warning: Negative gap ({gap_duration:.3f}s) before {segment_id} - segments may overlap")
            
            # Look for the audio file (match the actual format from the generator)
            # The generator creates files with hyphens in duration like "3-226" instead of "03.226"
            duration_formatted = f"{segment['duration']:.3f}".replace('.', '-')
            audio_filename = f"{segment_id}_{speaker}_{start_time_str}_{end_time_str}_{duration_formatted}.wav"
            audio_path = os.path.join(audio_dir, audio_filename)
            
            if os.path.exists(audio_path):
                # Convert to absolute path for FFmpeg compatibility
                absolute_audio_path = os.path.abspath(audio_path)
                temp_files.append(absolute_audio_path)
                print(f"  Added: {audio_filename} (starts at {start_time:.3f}s)")
            else:
                print(f"  Warning: Audio file not found: {audio_path}")
            
            previous_end_time = end_time
        
        # Create file list for ffmpeg concatenation
        file_list_path = os.path.join(temp_dir, "audio_filelist.txt")
        
        with open(file_list_path, 'w') as f:
            for temp_file in temp_files:
                f.write(f"file '{temp_file}'\n")
        
        # Debug: Show what files are being concatenated
        print(f"Files to concatenate ({len(temp_files)} total):")
        with open(file_list_path, 'r') as f:
            content = f.read()
            print(content[:500] + "..." if len(content) > 500 else content)
        
        # Concatenate audio files using ffmpeg
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", file_list_path,
            "-c:a", "pcm_s16le",
            "-ar", "22050",
            "-y",
            output_path
        ]
        
        try:
            print("Concatenating audio segments with timing gaps...")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("Audio concatenation completed successfully!")
            if result.stderr:
                print(f"FFmpeg warnings: {result.stderr}")
            
            # Get audio duration
            duration_cmd = [
                "ffprobe",
                "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                output_path
            ]
            
            try:
                duration_result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
                duration = float(duration_result.stdout.strip())
                original_duration = segments[-1]['end_time'] if segments else 0
                print(f"Final audio duration: {duration:.2f} seconds (original: {original_duration:.2f}s)")
            except:
                print("Could not determine final audio duration")
            
        except subprocess.CalledProcessError as e:
            print(f"Error concatenating audio: {e}")
            print(f"FFmpeg stderr: {e.stderr}")
            return False
    
    return True


def create_audio_summary(transcription, output_path):
    """Create a summary of the merged audio."""
    
    segments = transcription['segments']
    total_duration = sum(seg['duration'] for seg in segments)
    
    # Calculate timing information
    timing_info = {
        "initial_silence": 0.5,
        "total_gaps": 0.0,
        "gap_details": []
    }
    
    previous_end_time = None
    for i, segment in enumerate(segments):
        start_time = segment['start_time']
        if previous_end_time is not None:
            gap_duration = start_time - previous_end_time
            if gap_duration > 0.1:
                timing_info["total_gaps"] += gap_duration
                timing_info["gap_details"].append({
                    "before_segment": segment['segment_id'],
                    "gap_duration": gap_duration
                })
        previous_end_time = segment['end_time']
    
    summary = {
        "output_file": output_path,
        "total_segments": len(segments),
        "total_audio_duration": total_duration,
        "timing_preservation": timing_info,
        "estimated_final_duration": 0.5 + total_duration + timing_info["total_gaps"],
        "source_language": transcription.get('source_language', 'en'),
        "target_language": transcription.get('target_language', 'unknown'),
        "segments": []
    }
    
    for segment in segments:
        summary['segments'].append({
            "segment_id": segment['segment_id'],
            "speaker": segment['speaker'],
            "start_time": segment['start_time'],
            "end_time": segment['end_time'],
            "duration": segment['duration'],
            "translated_text": segment['translated_transcription']
        })
    
    summary_path = os.path.join(os.path.dirname(output_path), "merged_audio_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Audio summary saved to: {summary_path}")
    print(f"Timing summary:")
    print(f"  - Initial silence: {timing_info['initial_silence']}s")
    print(f"  - Total gaps added: {timing_info['total_gaps']:.3f}s")
    print(f"  - Number of gaps: {len(timing_info['gap_details'])}")
    print(f"  - Estimated final duration: {summary['estimated_final_duration']:.3f}s")
    
    return summary_path


def main():
    parser = argparse.ArgumentParser(description="Build merged audio from translated segments")
    parser.add_argument("--transcription", required=True,
                       help="Path to translated transcription JSON file")
    parser.add_argument("--audio-dir", required=True,
                       help="Directory containing translated audio segments")
    parser.add_argument("--output", 
                       default="audio_export/merged_translated_audio.wav",
                       help="Output path for merged audio file (default: audio_export/merged_translated_audio.wav)")
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.transcription):
        print(f"Error: Transcription file not found: {args.transcription}")
        sys.exit(1)
    
    if not os.path.exists(args.audio_dir):
        print(f"Error: Audio directory not found: {args.audio_dir}")
        sys.exit(1)
    
    # Load transcription
    transcription = load_transcription(args.transcription)
    if not transcription:
        sys.exit(1)
    
    # Build merged audio
    success = build_merged_audio(transcription, args.audio_dir, args.output)
    
    if not success:
        print("Audio building failed!")
        sys.exit(1)
    
    # Create summary
    create_audio_summary(transcription, args.output)
    
    print("Audio building completed successfully!")


if __name__ == "__main__":
    main() 