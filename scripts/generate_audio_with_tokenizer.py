#!/usr/bin/env python3
"""
Translated audio generation script using XTTS-v2 with Whisper verification.
Generates translated audio from translated transcriptions.
"""

import os
import sys
import json
import argparse
import whisper
import torch
from TTS.api import TTS
from pathlib import Path


def load_translations(input_file):
    """Load translated transcriptions from JSON file."""
    if not os.path.exists(input_file):
        print(f"Error: Translation file not found: {input_file}")
        return None
    
    with open(input_file, 'r') as f:
        return json.load(f)


def load_speaker_samples(speaker_dir):
    """Load speaker samples for voice cloning."""
    speaker_samples = {}
    
    if not os.path.exists(speaker_dir):
        print(f"Warning: Speaker directory not found: {speaker_dir}")
        return speaker_samples
    
    # Look for speaker sample files
    for file in os.listdir(speaker_dir):
        if file.endswith('_sample.wav'):
            speaker_id = file.replace('_sample.wav', '')
            speaker_samples[speaker_id] = os.path.join(speaker_dir, file)
    
    print(f"Found {len(speaker_samples)} speaker samples")
    return speaker_samples


def verify_transcription_with_whisper(audio_path, expected_text, language="en"):
    """Verify generated audio transcription using Whisper."""
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, language=language)
        transcribed_text = result['text'].strip().lower()
        expected_text_lower = expected_text.lower()
        
        # Simple similarity check (can be improved)
        similarity = len(set(transcribed_text.split()) & set(expected_text_lower.split())) / max(len(expected_text_lower.split()), 1)
        confidence = result.get('avg_logprob', 0.0)
        
        return {
            'similarity': similarity,
            'confidence': confidence,
            'transcribed': transcribed_text,
            'expected': expected_text_lower
        }
    except Exception as e:
        print(f"  Warning: Whisper verification failed: {e}")
        return None


def generate_translated_audio(translations, output_dir, language, speaker_dir, max_retries=5, confidence_threshold=85.0):
    """Generate translated audio using XTTS-v2."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load speaker samples
    speaker_samples = load_speaker_samples(speaker_dir)
    
    # Initialize XTTS model
    print("Loading XTTS-v2 model...")
    try:
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    except Exception as e:
        print(f"Error loading XTTS model: {e}")
        return False
    
    print(f"Generating audio for {len(translations['segments'])} segments...")
    
    generated_segments = []
    
    for i, segment in enumerate(translations['segments']):
        segment_id = segment['segment_id']
        speaker = segment['speaker']
        translated_text = segment['translated_transcription']
        
        print(f"Generating audio for segment {i+1}/{len(translations['segments'])}: {segment_id}")
        print(f"  Speaker: {speaker}")
        print(f"  Text: {translated_text}")
        
        # Find speaker sample
        speaker_sample = None
        for sample_id, sample_path in speaker_samples.items():
            if sample_id in speaker or speaker in sample_id:
                speaker_sample = sample_path
                break
        
        if not speaker_sample:
            print(f"  Warning: No speaker sample found for {speaker}")
            continue
        
        # Generate audio with retries
        success = False
        for attempt in range(max_retries):
            try:
                output_filename = f"{segment_id}_{speaker}_{segment['start_time_str']}_{segment['end_time_str']}_{segment['duration_str']}.wav"
                output_path = os.path.join(output_dir, output_filename)
                
                # Generate audio using XTTS
                tts.tts_to_file(
                    text=translated_text,
                    speaker_wav=speaker_sample,
                    language=language,
                    file_path=output_path
                )
                
                # Verify with Whisper
                verification = verify_transcription_with_whisper(output_path, translated_text, language)
                
                if verification and verification['similarity'] > 0.7:
                    print(f"  Audio generated successfully (attempt {attempt + 1})")
                    print(f"  Whisper similarity: {verification['similarity']:.2f}")
                    
                    generated_segments.append({
                        "segment_id": segment_id,
                        "speaker": speaker,
                        "audio_file": output_filename,
                        "translated_text": translated_text,
                        "whisper_verification": verification,
                        "attempts": attempt + 1
                    })
                    
                    success = True
                    break
                else:
                    print(f"  Whisper verification failed (attempt {attempt + 1})")
                    if verification:
                        print(f"    Similarity: {verification['similarity']:.2f}")
                        print(f"    Expected: {verification['expected']}")
                        print(f"    Got: {verification['transcribed']}")
                    
                    # Remove failed attempt
                    if os.path.exists(output_path):
                        os.remove(output_path)
                        
            except Exception as e:
                print(f"  Error generating audio (attempt {attempt + 1}): {e}")
                continue
        
        if not success:
            print(f"  Failed to generate audio for {segment_id} after {max_retries} attempts")
    
    # Save generation report
    report_data = {
        "model_used": "XTTS-v2",
        "language": language,
        "total_segments": len(translations['segments']),
        "successful_generations": len(generated_segments),
        "confidence_threshold": confidence_threshold,
        "max_retries": max_retries,
        "generated_segments": generated_segments
    }
    
    report_path = os.path.join(output_dir, "generation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # Copy translated transcription to output directory
    transcription_path = os.path.join(output_dir, "translated_transcription.json")
    with open(transcription_path, 'w') as f:
        json.dump(translations, f, indent=2)
    
    print(f"\nAudio generation completed!")
    print(f"Successful generations: {len(generated_segments)}/{len(translations['segments'])}")
    print(f"Report saved to: {report_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate translated audio using XTTS-v2")
    parser.add_argument("--input", default="translated_transcription/translated_transcription.json",
                       help="Path to translated transcriptions JSON file")
    parser.add_argument("--output", default="translated_segments",
                       help="Output directory for generated audio")
    parser.add_argument("--language", required=True,
                       help="Target language for TTS")
    parser.add_argument("--speaker-dir", default="speaker_segments",
                       help="Directory containing speaker samples")
    parser.add_argument("--max-retries", type=int, default=5,
                       help="Maximum retry attempts for audio generation")
    parser.add_argument("--confidence-threshold", type=float, default=85.0,
                       help="Confidence threshold for Whisper verification")
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Load translations
    translations = load_translations(args.input)
    if not translations:
        sys.exit(1)
    
    # Generate audio
    success = generate_translated_audio(
        translations, 
        args.output, 
        args.language, 
        args.speaker_dir,
        args.max_retries,
        args.confidence_threshold
    )
    
    if not success:
        print("Audio generation failed!")
        sys.exit(1)
    else:
        print("Audio generation completed successfully!")


if __name__ == "__main__":
    main() 