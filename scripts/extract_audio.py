#!/usr/bin/env python3
"""
Audio extraction script for video files.
Extracts audio from video files using ffmpeg.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def extract_audio(video_path, output_dir="audio_export"):
    """Extract audio from video file using ffmpeg."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video filename without extension
    video_name = Path(video_path).stem
    output_path = os.path.join(output_dir, f"{video_name}_audio.wav")
    
    print(f"Extracting audio from: {video_path}")
    print(f"Output path: {output_path}")
    
    # FFmpeg command to extract audio
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # PCM 16-bit
        "-ar", "16000",  # Sample rate 16kHz
        "-ac", "1",  # Mono
        "-y",  # Overwrite output file
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Audio extraction completed successfully!")
        print(f"Output file: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        print(f"FFmpeg stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg.")
        return None


def main():
    parser = argparse.ArgumentParser(description="Extract audio from video file")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--output-dir", default="audio_export", 
                       help="Output directory for extracted audio")
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Input video file not found: {args.video_path}")
        sys.exit(1)
    
    # Extract audio
    result = extract_audio(args.video_path, args.output_dir)
    
    if result is None:
        print("Audio extraction failed!")
        sys.exit(1)
    else:
        print("Audio extraction completed successfully!")


if __name__ == "__main__":
    main() 