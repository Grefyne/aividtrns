#!/usr/bin/env python3
"""
Final audio mixing script.
Combines translated audio with background music to create final output.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path


def find_audio_files():
    """Find the required audio files for mixing."""
    
    # Look for translated audio
    translated_audio_path = "translated_audio/merged_translated_audio.wav"
    if not os.path.exists(translated_audio_path):
        print(f"Error: Translated audio not found: {translated_audio_path}")
        return None, None
    
    # Look for background music (vocal removal output)
    background_music_path = None
    
    # Check vocal_removal_out directory
    vocal_removal_dir = "vocal_removal_out"
    if os.path.exists(vocal_removal_dir):
        for file in os.listdir(vocal_removal_dir):
            if file.endswith("_vocrem.wav"):
                background_music_path = os.path.join(vocal_removal_dir, file)
                break
    
    if not background_music_path:
        print("Warning: Background music not found. Will use original audio as background.")
        # Use original audio as fallback
        audio_export_dir = "audio_export"
        if os.path.exists(audio_export_dir):
            for file in os.listdir(audio_export_dir):
                if file.endswith("_audio.wav") and not file.endswith("_vocrem.wav"):
                    background_music_path = os.path.join(audio_export_dir, file)
                    break
    
    return translated_audio_path, background_music_path


def mix_audio(translated_audio_path, background_music_path, output_path="audio_export/final_audio.wav"):
    """Mix translated audio with background music."""
    
    print(f"Mixing translated audio: {translated_audio_path}")
    if background_music_path:
        print(f"With background music: {background_music_path}")
    else:
        print("No background music found, using translated audio only")
    
    print(f"Output path: {output_path}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if background_music_path:
        # Mix translated audio with background music
        cmd = [
            "ffmpeg",
            "-i", translated_audio_path,
            "-i", background_music_path,
            "-filter_complex", "[0:a]volume=1.0[translated];[1:a]volume=0.3[background];[translated][background]amix=inputs=2:duration=longest",
            "-y",
            output_path
        ]
    else:
        # Just copy translated audio
        cmd = [
            "ffmpeg",
            "-i", translated_audio_path,
            "-c", "copy",
            "-y",
            output_path
        ]
    
    try:
        print("Mixing audio...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Audio mixing completed successfully!")
        
        # Get final audio duration
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
        
        return output_path
        
    except subprocess.CalledProcessError as e:
        print(f"Error mixing audio: {e}")
        print(f"FFmpeg stderr: {e.stderr}")
        return None


def create_final_audio_summary(translated_audio_path, background_music_path, output_path):
    """Create a summary of the final audio mixing process."""
    
    summary = {
        "translated_audio": translated_audio_path,
        "background_music": background_music_path,
        "final_audio": output_path,
        "process": "Audio mixing with FFmpeg",
        "description": "Combined translated audio with background music"
    }
    
    summary_path = os.path.join(os.path.dirname(output_path), "final_audio_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Final audio summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Create final audio mix")
    parser.add_argument("--output", default="audio_export/final_audio.wav",
                       help="Output path for final audio file")
    args = parser.parse_args()
    
    # Find audio files
    translated_audio_path, background_music_path = find_audio_files()
    
    if not translated_audio_path:
        print("Error: Required audio files not found!")
        sys.exit(1)
    
    # Mix audio
    result = mix_audio(translated_audio_path, background_music_path, args.output)
    
    if result is None:
        print("Audio mixing failed!")
        sys.exit(1)
    else:
        # Create summary
        create_final_audio_summary(translated_audio_path, background_music_path, result)
        print("Final audio mixing completed successfully!")


if __name__ == "__main__":
    main() 