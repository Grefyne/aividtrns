#!/usr/bin/env python3
"""
Speaker audio segmentation script using pyannote.audio.
Performs speaker diarization to segment audio by speaker.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import torch
from gpu_utils import get_device, setup_multi_gpu_processing, clear_gpu_cache


def load_hf_token():
    """Load HuggingFace token from config file."""
    token_path = "config/hf_token.txt"
    if os.path.exists(token_path):
        with open(token_path, 'r') as f:
            return f.read().strip()
    else:
        print(f"Warning: HuggingFace token file not found at {token_path}")
        print("Please create config/hf_token.txt with your HuggingFace token")
        return None


def perform_speaker_segmentation(audio_path, output_dir="speaker_segments", gpu_id=None):
    """Perform speaker diarization using pyannote.audio."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load HuggingFace token
    hf_token = load_hf_token()
    if not hf_token:
        print("Error: HuggingFace token required for pyannote.audio")
        return False
    
    print(f"Performing speaker segmentation on: {audio_path}")
    print(f"Output directory: {output_dir}")
    
    # Setup GPU processing
    available_gpus = setup_multi_gpu_processing()
    if available_gpus:
        if gpu_id is None:
            gpu_id = available_gpus[0]  # Use first available GPU
        device = get_device(gpu_id)
        print(f"Using GPU {gpu_id} for processing")
    else:
        device = torch.device("cpu")
        print("Using CPU for processing")
    
    try:
        # Initialize pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        # Move to appropriate device
        pipeline = pipeline.to(device)
        
        # Perform diarization
        with ProgressHook() as hook:
            diarization = pipeline(audio_path, hook=hook)
        
        # Extract raw segments first
        raw_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_time = turn.start
            end_time = turn.end
            duration = end_time - start_time
            
            # Skip very short segments (less than 0.5 seconds)
            if duration < 0.5:
                continue
            
            raw_segments.append({
                "speaker": speaker,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration
            })
        
        # Merge consecutive segments from the same speaker
        merged_segments = []
        segment_counter = 1
        
        if raw_segments:
            current_segment = raw_segments[0].copy()
            current_segment["segment_id"] = f"segment{segment_counter:03d}"
            
            for next_segment in raw_segments[1:]:
                # If same speaker and segments are close (within 1 second gap)
                if (next_segment["speaker"] == current_segment["speaker"] and 
                    next_segment["start_time"] - current_segment["end_time"] <= 1.0):
                    # Merge segments
                    current_segment["end_time"] = next_segment["end_time"]
                    current_segment["duration"] = current_segment["end_time"] - current_segment["start_time"]
                else:
                    # Different speaker or gap too large, save current segment and start new one
                    # Add formatted time strings
                    current_segment["start_time_str"] = f"{int(current_segment['start_time']//3600):02d}-{int((current_segment['start_time']%3600)//60):02d}-{current_segment['start_time']%60:06.3f}"
                    current_segment["end_time_str"] = f"{int(current_segment['end_time']//3600):02d}-{int((current_segment['end_time']%3600)//60):02d}-{current_segment['end_time']%60:06.3f}"
                    current_segment["duration_str"] = f"{current_segment['duration']:06.3f}"
                    
                    merged_segments.append(current_segment)
                    
                    # Start new segment
                    segment_counter += 1
                    current_segment = next_segment.copy()
                    current_segment["segment_id"] = f"segment{segment_counter:03d}"
            
            # Add the last segment
            current_segment["start_time_str"] = f"{int(current_segment['start_time']//3600):02d}-{int((current_segment['start_time']%3600)//60):02d}-{current_segment['start_time']%60:06.3f}"
            current_segment["end_time_str"] = f"{int(current_segment['end_time']//3600):02d}-{int((current_segment['end_time']%3600)//60):02d}-{current_segment['end_time']%60:06.3f}"
            current_segment["duration_str"] = f"{current_segment['duration']:06.3f}"
            merged_segments.append(current_segment)
        
        # Save segments summary
        summary = {
            "audio_file": audio_path,
            "total_segments": len(merged_segments),
            "speakers": list(set(seg["speaker"] for seg in merged_segments)),
            "segments": merged_segments
        }
        
        summary_path = os.path.join(output_dir, "segments_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Segmentation completed!")
        print(f"Original segments: {len(raw_segments)}")
        print(f"Merged segments: {len(merged_segments)}")
        print(f"Speakers detected: {summary['speakers']}")
        print(f"Summary saved to: {summary_path}")
        
        # Extract individual audio segments
        extract_audio_segments(audio_path, merged_segments, output_dir)
        
        # Clear GPU cache
        if available_gpus:
            clear_gpu_cache(gpu_id)
        
        return True
        
    except Exception as e:
        print(f"Error during speaker segmentation: {e}")
        # Clear GPU cache on error
        if available_gpus:
            clear_gpu_cache(gpu_id)
        return False


def extract_audio_segments(audio_path, segments, output_dir):
    """Extract individual audio segments using ffmpeg."""
    
    print("Extracting individual audio segments...")
    
    for segment in segments:
        segment_id = segment["segment_id"]
        speaker = segment["speaker"]
        start_time = segment["start_time"]
        end_time = segment["end_time"]
        
        # Create filename
        filename = f"{segment_id}_{speaker}_{segment['start_time_str']}_{segment['end_time_str']}_{segment['duration_str']}.wav"
        output_path = os.path.join(output_dir, filename)
        
        # FFmpeg command to extract segment
        cmd = [
            "ffmpeg",
            "-i", audio_path,
            "-ss", str(start_time),
            "-t", str(end_time - start_time),
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y",
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"  Extracted: {filename}")
        except subprocess.CalledProcessError as e:
            print(f"  Error extracting {filename}: {e}")
    
    print("Audio segment extraction completed!")


def main():
    parser = argparse.ArgumentParser(description="Perform speaker audio segmentation")
    parser.add_argument("audio_path", help="Path to input audio file")
    parser.add_argument("--output-dir", default="speaker_segments", 
                       help="Output directory for segments")
    parser.add_argument("--gpu-id", type=int, default=None,
                       help="Specific GPU ID to use (default: auto-select)")
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.audio_path):
        print(f"Error: Input audio file not found: {args.audio_path}")
        sys.exit(1)
    
    # Perform segmentation
    success = perform_speaker_segmentation(args.audio_path, args.output_dir, args.gpu_id)
    
    if not success:
        print("Speaker segmentation failed!")
        sys.exit(1)
    else:
        print("Speaker segmentation completed successfully!")


if __name__ == "__main__":
    main() 