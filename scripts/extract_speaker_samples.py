#!/usr/bin/env python3
"""
Speaker sample extraction script.
Extracts individual speaker samples from audio segments for voice cloning.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict


def load_segments_summary(summary_path):
    """Load segments summary from JSON file."""
    if not os.path.exists(summary_path):
        print(f"Error: Segments summary not found: {summary_path}")
        return None
    
    with open(summary_path, 'r') as f:
        return json.load(f)


def extract_speaker_samples(summary_path, audio_file, output_dir="speaker_samples"):
    """Extract individual speaker samples from audio segments."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load segments summary
    summary = load_segments_summary(summary_path)
    if not summary:
        return False
    
    print(f"Extracting speaker samples from: {audio_file}")
    print(f"Output directory: {output_dir}")
    
    # Group segments by speaker
    speaker_segments = defaultdict(list)
    for segment in summary['segments']:
        speaker_segments[segment['speaker']].append(segment)
    
    print(f"Found {len(speaker_segments)} speakers")
    
    extracted_samples = []
    
    for speaker, segments in speaker_segments.items():
        print(f"\nProcessing speaker: {speaker}")
        print(f"  Total segments: {len(segments)}")
        
        # Sort segments by start time
        segments.sort(key=lambda x: x['start_time'])
        
        # Calculate total duration for this speaker
        total_duration = sum(seg['duration'] for seg in segments)
        print(f"  Total duration: {total_duration:.2f} seconds")
        
        # Create speaker sample filename
        sample_filename = f"speaker{speaker.split('_')[-1]}_sample.wav"
        sample_path = os.path.join(output_dir, sample_filename)
        
        # If we have multiple segments, concatenate them
        if len(segments) > 1:
            print(f"  Concatenating {len(segments)} segments...")
            
            # Create a file list for ffmpeg concatenation
            file_list_path = os.path.join(output_dir, f"{speaker}_filelist.txt")
            
            with open(file_list_path, 'w') as f:
                for segment in segments:
                    # Find the segment audio file
                    segment_filename = f"{segment['segment_id']}_{speaker}_{segment['start_time_str']}_{segment['end_time_str']}_{segment['duration_str']}.wav"
                    segment_path = os.path.join("speaker_segments", segment_filename)
                    
                    if os.path.exists(segment_path):
                        # Use absolute path for ffmpeg
                        abs_segment_path = os.path.abspath(segment_path)
                        f.write(f"file '{abs_segment_path}'\n")
                    else:
                        print(f"    Warning: Segment file not found: {segment_path}")
            
            # Concatenate using ffmpeg
            cmd = [
                "ffmpeg",
                "-f", "concat",
                "-safe", "0",
                "-i", file_list_path,
                "-c", "copy",
                "-y",
                sample_path
            ]
            
            try:
                subprocess.run(cmd, capture_output=True, check=True)
                print(f"  Created sample: {sample_filename}")
                
                # Clean up file list
                os.remove(file_list_path)
                
                extracted_samples.append({
                    "speaker": speaker,
                    "sample_file": sample_filename,
                    "total_segments": len(segments),
                    "total_duration": total_duration
                })
                
            except subprocess.CalledProcessError as e:
                print(f"  Error creating sample for {speaker}: {e}")
                continue
                
        else:
            # Single segment, just copy it
            segment = segments[0]
            segment_filename = f"{segment['segment_id']}_{speaker}_{segment['start_time_str']}_{segment['end_time_str']}_{segment['duration_str']}.wav"
            segment_path = os.path.join("speaker_segments", segment_filename)
            
            if os.path.exists(segment_path):
                cmd = ["cp", segment_path, sample_path]
                try:
                    subprocess.run(cmd, check=True)
                    print(f"  Copied single segment: {sample_filename}")
                    
                    extracted_samples.append({
                        "speaker": speaker,
                        "sample_file": sample_filename,
                        "total_segments": 1,
                        "total_duration": segment['duration']
                    })
                    
                except subprocess.CalledProcessError as e:
                    print(f"  Error copying segment for {speaker}: {e}")
                    continue
            else:
                print(f"  Warning: Segment file not found: {segment_path}")
    
    # Save extraction summary
    summary_data = {
        "audio_file": audio_file,
        "total_speakers": len(extracted_samples),
        "samples": extracted_samples
    }
    
    summary_path = os.path.join(output_dir, "extraction_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nSpeaker sample extraction completed!")
    print(f"Extracted samples: {len(extracted_samples)}")
    print(f"Summary saved to: {summary_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Extract speaker samples from audio segments")
    parser.add_argument("summary_path", help="Path to segments summary JSON file")
    parser.add_argument("audio_file", help="Path to original audio file")
    parser.add_argument("--output-dir", default="speaker_samples", 
                       help="Output directory for speaker samples")
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.summary_path):
        print(f"Error: Summary file not found: {args.summary_path}")
        sys.exit(1)
    
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    # Extract speaker samples
    success = extract_speaker_samples(args.summary_path, args.audio_file, args.output_dir)
    
    if not success:
        print("Speaker sample extraction failed!")
        sys.exit(1)
    else:
        print("Speaker sample extraction completed successfully!")


if __name__ == "__main__":
    main() 