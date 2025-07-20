#!/usr/bin/env python3
"""
Audio building script to merge translated audio segments into final audio file.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path


def load_transcription(transcription_path):
    """Load translated transcription from JSON file."""
    if not os.path.exists(transcription_path):
        print(f"Error: Transcription file not found: {transcription_path}")
        return None
    
    with open(transcription_path, 'r') as f:
        return json.load(f)


def build_merged_audio(transcription, audio_dir, output_path):
    """Build merged audio from translated segments."""
    
    print(f"Building merged audio from: {audio_dir}")
    print(f"Output path: {output_path}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sort segments by start time
    segments = sorted(transcription['segments'], key=lambda x: x['start_time'])
    
    print(f"Processing {len(segments)} segments...")
    
    # Create a file list for ffmpeg concatenation
    file_list_path = "temp_audio_filelist.txt"
    
    with open(file_list_path, 'w') as f:
        for segment in segments:
            segment_id = segment['segment_id']
            speaker = segment['speaker']
            start_time_str = segment['start_time_str']
            end_time_str = segment['end_time_str']
            duration_str = segment['duration_str']
            
            # Look for the audio file
            audio_filename = f"{segment_id}_{speaker}_{start_time_str}_{end_time_str}_{duration_str}.wav"
            audio_path = os.path.join(audio_dir, audio_filename)
            
            if os.path.exists(audio_path):
                f.write(f"file '{audio_path}'\n")
                print(f"  Added: {audio_filename}")
            else:
                print(f"  Warning: Audio file not found: {audio_path}")
    
    # Concatenate audio files using ffmpeg
    cmd = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", file_list_path,
        "-c", "copy",
        "-y",
        output_path
    ]
    
    try:
        print("Concatenating audio segments...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Audio concatenation completed successfully!")
        
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
            print(f"Final audio duration: {duration:.2f} seconds")
        except:
            print("Could not determine final audio duration")
        
    except subprocess.CalledProcessError as e:
        print(f"Error concatenating audio: {e}")
        print(f"FFmpeg stderr: {e.stderr}")
        return False
    finally:
        # Clean up temporary file
        if os.path.exists(file_list_path):
            os.remove(file_list_path)
    
    return True


def create_audio_summary(transcription, output_path):
    """Create a summary of the merged audio."""
    
    segments = transcription['segments']
    total_duration = sum(seg['duration'] for seg in segments)
    
    summary = {
        "output_file": output_path,
        "total_segments": len(segments),
        "total_duration": total_duration,
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
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Audio summary saved to: {summary_path}")
    return summary_path


def main():
    parser = argparse.ArgumentParser(description="Build merged audio from translated segments")
    parser.add_argument("--transcription", required=True,
                       help="Path to translated transcription JSON file")
    parser.add_argument("--audio-dir", required=True,
                       help="Directory containing translated audio segments")
    parser.add_argument("--output", required=True,
                       help="Output path for merged audio file")
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