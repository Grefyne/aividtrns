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
from pathlib import Path


def load_segments_summary(segments_dir):
    """Load segments summary from JSON file."""
    summary_path = os.path.join(segments_dir, "segments_summary.json")
    if not os.path.exists(summary_path):
        print(f"Error: Segments summary not found: {summary_path}")
        return None
    
    with open(summary_path, 'r') as f:
        return json.load(f)


def transcribe_segments(segments_dir, output_dir, model_name="large"):
    """Transcribe all audio segments using Whisper."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load segments summary
    summary = load_segments_summary(segments_dir)
    if not summary:
        return False
    
    print(f"Loading Whisper model: {model_name}")
    try:
        model = whisper.load_model(model_name)
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return False
    
    print(f"Transcribing {len(summary['segments'])} segments...")
    
    transcribed_segments = []
    
    for i, segment in enumerate(summary['segments']):
        segment_id = segment['segment_id']
        speaker = segment['speaker']
        
        # Find corresponding audio file
        audio_filename = f"{segment_id}_{speaker}_{segment['start_time_str']}_{segment['end_time_str']}_{segment['duration_str']}.wav"
        audio_path = os.path.join(segments_dir, audio_filename)
        
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found: {audio_path}")
            continue
        
        print(f"Transcribing segment {i+1}/{len(summary['segments'])}: {segment_id}")
        
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
            
            transcribed_segments.append(transcription_entry)
            print(f"  Transcription: {transcription_entry['transcription']}")
            
        except Exception as e:
            print(f"  Error transcribing {segment_id}: {e}")
            continue
    
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
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio segments using Whisper")
    parser.add_argument("--segments_dir", default="speaker_segments", 
                       help="Directory containing audio segments")
    parser.add_argument("--output_dir", default="transcribed_segments", 
                       help="Output directory for transcriptions")
    parser.add_argument("--model", default="large", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model to use")
    args = parser.parse_args()
    
    # Check if segments directory exists
    if not os.path.exists(args.segments_dir):
        print(f"Error: Segments directory not found: {args.segments_dir}")
        sys.exit(1)
    
    # Perform transcription
    success = transcribe_segments(args.segments_dir, args.output_dir, args.model)
    
    if not success:
        print("Transcription failed!")
        sys.exit(1)
    else:
        print("Transcription completed successfully!")


if __name__ == "__main__":
    main() 