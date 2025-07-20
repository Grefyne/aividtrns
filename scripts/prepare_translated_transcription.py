#!/usr/bin/env python3
"""
Script to convert generation report and transcribed segments into format expected by build_audio.py
"""

import json
import sys
import os


def load_json(file_path):
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_translated_transcription(transcribed_segments_path, generation_report_path, output_path):
    """Create translated transcription JSON for build_audio.py"""
    
    # Load both files
    transcribed = load_json(transcribed_segments_path)
    generation = load_json(generation_report_path)
    
    # Create mapping from segment_id to generation data
    generation_map = {}
    for segment in generation['segments']:
        generation_map[segment['segment_id']] = segment
    
    # Build the translated transcription
    translated_transcription = {
        "source_language": "en",
        "target_language": "en",  # Since it's the same language in this case
        "total_segments": len(transcribed['segments']),
        "segments": []
    }
    
    for segment in transcribed['segments']:
        segment_id = segment['segment_id']
        
        # Get translated text from generation report
        translated_text = ""
        if segment_id in generation_map:
            translated_text = generation_map[segment_id]['combined_text']
        else:
            # Fallback to original transcription
            translated_text = segment['transcription']
        
        # Create translated segment
        translated_segment = {
            "segment_id": segment_id,
            "speaker": segment['speaker'],
            "start_time": segment['start_time'],
            "end_time": segment['end_time'],
            "duration": segment['duration'],
            "start_time_str": segment['start_time_str'],
            "end_time_str": segment['end_time_str'],
            "duration_str": segment['duration_str'],
            "translated_transcription": translated_text
        }
        
        translated_transcription['segments'].append(translated_segment)
    
    # Save the translated transcription
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(translated_transcription, f, indent=2, ensure_ascii=False)
    
    print(f"Created translated transcription: {output_path}")
    return output_path


def main():
    if len(sys.argv) != 4:
        print("Usage: python prepare_translated_transcription.py <transcribed_segments.json> <generation_report.json> <output.json>")
        sys.exit(1)
    
    transcribed_segments_path = sys.argv[1]
    generation_report_path = sys.argv[2]
    output_path = sys.argv[3]
    
    # Check if input files exist
    if not os.path.exists(transcribed_segments_path):
        print(f"Error: Transcribed segments file not found: {transcribed_segments_path}")
        sys.exit(1)
    
    if not os.path.exists(generation_report_path):
        print(f"Error: Generation report file not found: {generation_report_path}")
        sys.exit(1)
    
    create_translated_transcription(transcribed_segments_path, generation_report_path, output_path)


if __name__ == "__main__":
    main() 