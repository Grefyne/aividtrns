#!/usr/bin/env python3
"""
Audio segment transcription script using Whisper.
Transcribes individual audio segments using OpenAI's Whisper model.
"""

import os
import sys
import json
import argparse
import whisper
import torch
from pathlib import Path
from gpu_utils import get_device, setup_multi_gpu_processing, clear_gpu_cache, parallel_process_with_gpus


def load_segments_summary(segments_dir):
    """Load segments summary from JSON file."""
    summary_path = os.path.join(segments_dir, "segments_summary.json")
    if not os.path.exists(summary_path):
        print(f"Error: Segments summary not found: {summary_path}")
        return None
    
    with open(summary_path, 'r') as f:
        return json.load(f)


def transcribe_segments(segments_dir, output_dir, model_name="large", gpu_id=None, use_parallel=False):
    """Transcribe all audio segments using Whisper."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load segments summary
    summary = load_segments_summary(segments_dir)
    if not summary:
        return False
    
    # Setup GPU processing
    available_gpus = setup_multi_gpu_processing()
    if available_gpus:
        if gpu_id is None:
            gpu_id = available_gpus[0]  # Use first available GPU
        device = get_device(gpu_id)
        print(f"Using GPU {gpu_id} for Whisper processing")
    else:
        device = torch.device("cpu")
        print("Using CPU for Whisper processing")
    
    print(f"Loading Whisper model: {model_name}")
    try:
        model = whisper.load_model(model_name)
        # Move model to device
        if hasattr(model, 'to'):
            model = model.to(device)
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return False
    
    print(f"Transcribing {len(summary['segments'])} segments...")
    
    if use_parallel and len(available_gpus) > 1:
        print(f"Using parallel processing across {len(available_gpus)} GPUs")
        transcribed_segments = parallel_process_with_gpus(
            lambda segment: _transcribe_single_segment(segment, segments_dir, model, device),
            summary['segments'],
            num_gpus=len(available_gpus)
        )
        # Filter out None results
        transcribed_segments = [seg for seg in transcribed_segments if seg is not None]
    else:
        transcribed_segments = []
        
        for i, segment in enumerate(summary['segments']):
            result = _transcribe_single_segment(segment, segments_dir, model, device, i+1, len(summary['segments']))
            if result:
                transcribed_segments.append(result)
    
    # Save transcriptions
    output_data = {
        "model_used": model_name,
        "total_segments": len(transcribed_segments),
        "segments": transcribed_segments
    }
    
    output_path = os.path.join(output_dir, "transcribed_segments.json")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nTranscription completed!")
    print(f"Transcribed segments: {len(transcribed_segments)}/{len(summary['segments'])}")
    print(f"Output saved to: {output_path}")
    
    # Clear GPU cache
    if available_gpus:
        clear_gpu_cache(gpu_id)
    
    return True


def _transcribe_single_segment(segment, segments_dir, model, device, segment_num=None, total_segments=None):
    """Transcribe a single audio segment."""
    segment_id = segment['segment_id']
    speaker = segment['speaker']
    
    # Find corresponding audio file
    audio_filename = f"{segment_id}_{speaker}_{segment['start_time_str']}_{segment['end_time_str']}_{segment['duration_str']}.wav"
    audio_path = os.path.join(segments_dir, audio_filename)
    
    if not os.path.exists(audio_path):
        print(f"Warning: Audio file not found: {audio_path}")
        return None
    
    if segment_num and total_segments:
        print(f"Transcribing segment {segment_num}/{total_segments}: {segment_id}")
    else:
        print(f"Transcribing segment: {segment_id}")
    
    try:
        # Transcribe audio
        result = model.transcribe(audio_path)
        
        # Create transcription entry
        transcription_entry = {
            "segment_id": segment_id,
            "speaker": speaker,
            "start_time": segment['start_time'],
            "end_time": segment['end_time'],
            "duration": segment['duration'],
            "start_time_str": segment['start_time_str'],
            "end_time_str": segment['end_time_str'],
            "duration_str": segment['duration_str'],
            "audio_file": audio_filename,
            "transcription": result['text'].strip(),
            "language": result.get('language', 'en'),
            "confidence": result.get('avg_logprob', 0.0)
        }
        
        if segment_num and total_segments:
            print(f"  Transcription: {transcription_entry['transcription']}")
        
        return transcription_entry
        
    except Exception as e:
        print(f"  Error transcribing {segment_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio segments using Whisper")
    parser.add_argument("--segments_dir", default="speaker_segments", 
                       help="Directory containing audio segments")
    parser.add_argument("--output_dir", default="transcribed_segments", 
                       help="Output directory for transcriptions")
    parser.add_argument("--model", default="large", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model to use")
    parser.add_argument("--gpu-id", type=int, default=None,
                       help="Specific GPU ID to use (default: auto-select)")
    parser.add_argument("--parallel", action="store_true",
                       help="Use parallel processing across multiple GPUs")
    args = parser.parse_args()
    
    # Check if segments directory exists
    if not os.path.exists(args.segments_dir):
        print(f"Error: Segments directory not found: {args.segments_dir}")
        sys.exit(1)
    
    # Perform transcription
    success = transcribe_segments(args.segments_dir, args.output_dir, args.model, args.gpu_id, args.parallel)
    
    if not success:
        print("Transcription failed!")
        sys.exit(1)
    else:
        print("Transcription completed successfully!")


if __name__ == "__main__":
    main() 