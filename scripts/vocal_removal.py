#!/usr/bin/env python3
"""
Vocal removal script using Spleeter.
Removes vocals from original audio to create background music.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def remove_vocals(audio_path, output_dir="vocal_removal_out"):
    """Remove vocals from audio using Spleeter."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get audio filename without extension
    audio_name = Path(audio_path).stem
    output_path = os.path.join(output_dir, f"{audio_name}_vocrem.wav")
    
    print(f"Removing vocals from: {audio_path}")
    print(f"Output path: {output_path}")
    
    # Spleeter command to separate vocals and accompaniment
    cmd = [
        "spleeter", "separate",
        "-p", "spleeter:2stems",  # Separate into vocals and accompaniment
        "-o", output_dir,
        audio_path
    ]
    
    try:
        print("Running Spleeter vocal separation...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Vocal separation completed successfully!")
        
        # Spleeter creates a subdirectory with the audio name
        spleeter_output_dir = os.path.join(output_dir, audio_name)
        accompaniment_path = os.path.join(spleeter_output_dir, "accompaniment.wav")
        
        if os.path.exists(accompaniment_path):
            # Move accompaniment to main output directory
            final_output_path = os.path.join(output_dir, f"{audio_name}_vocrem.wav")
            subprocess.run(["mv", accompaniment_path, final_output_path], check=True)
            
            # Clean up spleeter output directory
            subprocess.run(["rm", "-rf", spleeter_output_dir], check=True)
            
            print(f"Vocal removal completed!")
            print(f"Background music saved to: {final_output_path}")
            
            # Get audio duration
            duration_cmd = [
                "ffprobe",
                "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                final_output_path
            ]
            
            try:
                duration_result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
                duration = float(duration_result.stdout.strip())
                print(f"Background music duration: {duration:.2f} seconds")
            except:
                print("Could not determine background music duration")
            
            return final_output_path
        else:
            print(f"Error: Accompaniment file not found: {accompaniment_path}")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Error removing vocals: {e}")
        print(f"Spleeter stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Error: Spleeter not found. Please install Spleeter: pip install spleeter")
        return None


def create_vocal_removal_summary(audio_path, output_path):
    """Create a summary of the vocal removal process."""
    
    summary = {
        "original_audio": audio_path,
        "background_music": output_path,
        "process": "Spleeter 2stems separation",
        "description": "Vocal removal using Spleeter to create background music"
    }
    
    summary_path = os.path.join(os.path.dirname(output_path), "vocal_removal_summary.json")
    with open(summary_path, 'w') as f:
        import json
        json.dump(summary, f, indent=2)
    
    print(f"Vocal removal summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Remove vocals from audio using Spleeter")
    parser.add_argument("audio_path", help="Path to input audio file")
    parser.add_argument("--output-dir", default="vocal_removal_out", 
                       help="Output directory for vocal removal")
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.audio_path):
        print(f"Error: Input audio file not found: {args.audio_path}")
        sys.exit(1)
    
    # Remove vocals
    result = remove_vocals(args.audio_path, args.output_dir)
    
    if result is None:
        print("Vocal removal failed!")
        sys.exit(1)
    else:
        # Create summary
        create_vocal_removal_summary(args.audio_path, result)
        print("Vocal removal completed successfully!")


if __name__ == "__main__":
    main() 