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
from gpu_utils import get_device, setup_multi_gpu_processing, clear_gpu_cache, parallel_process_with_gpus


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


def verify_transcription_with_whisper(audio_path, expected_text, language="en", device=None):
    """Verify generated audio transcription using Whisper."""
    try:
        model = whisper.load_model("base")
        if device and hasattr(model, 'to'):
            model = model.to(device)
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


def generate_translated_audio(translations, output_dir, language, speaker_dir, max_retries=5, confidence_threshold=85.0, gpu_id=None, use_parallel=False):
    """Generate translated audio using XTTS-v2."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load speaker samples
    speaker_samples = load_speaker_samples(speaker_dir)
    
    # Setup GPU processing
    if gpu_id is None:
        # Automatically select the best GPU with most free VRAM
        available_gpus = setup_multi_gpu_processing()
        if available_gpus:
            from gpu_utils import select_best_gpu
            gpu_id = select_best_gpu()
            if gpu_id is not None:
                device = get_device(gpu_id)
                print(f"Automatically selected GPU {gpu_id} for XTTS processing")
            else:
                device = torch.device("cpu")
                print("No suitable GPU found, using CPU for XTTS processing")
        else:
            device = torch.device("cpu")
            print("No GPUs available, using CPU for XTTS processing")
    else:
        device = get_device(gpu_id)
        print(f"Using specified GPU {gpu_id} for XTTS processing")
    
    # Initialize XTTS model
    print("Loading XTTS-v2 model...")
    try:
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        # XTTS automatically uses GPU if available, but we can set device explicitly
        if hasattr(tts, 'to') and device.type == 'cuda':
            tts = tts.to(device)
    except Exception as e:
        print(f"Error loading XTTS model: {e}")
        return False
    
    print(f"Processing {len(translations['segments'])} segments...")
    
    generated_segments = []
    skipped_segments = []
    
    for i, segment in enumerate(translations['segments']):
        segment_id = segment['segment_id']
        speaker = segment['speaker']
        translated_text = segment['translated_transcription']
        duration = segment.get('duration', 0)
        
        print(f"Processing segment {i+1}/{len(translations['segments'])}: {segment_id}")
        print(f"  Speaker: {speaker}")
        print(f"  Duration: {duration:.3f}s")
        print(f"  Text: {translated_text}")
        
        # Skip segments shorter than 0.5 seconds
        if duration < 0.5:
            print(f"  Skipping segment {segment_id} - too short ({duration:.3f}s < 0.5s)")
            skipped_segments.append({
                "segment_id": segment_id,
                "duration": duration,
                "reason": "too_short"
            })
            continue
        
        # Find speaker sample
        speaker_sample = None
        for sample_id, sample_path in speaker_samples.items():
            # Match SPEAKER_00 with speaker00, SPEAKER_01 with speaker01, etc.
            if sample_id in speaker.lower() or speaker.lower().replace('_', '') in sample_id:
                speaker_sample = sample_path
                break
        
        if not speaker_sample:
            print(f"  Warning: No speaker sample found for {speaker}")
            continue
        
        print(f"  Generating audio for segment {segment_id}...")
        
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
                verification = verify_transcription_with_whisper(output_path, translated_text, language, device)
                
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
        "skipped_segments": len(skipped_segments),
        "confidence_threshold": confidence_threshold,
        "max_retries": max_retries,
        "generated_segments": generated_segments,
        "skipped_segments": skipped_segments
    }
    
    report_path = os.path.join(output_dir, "generation_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    # Copy translated transcription to output directory
    transcription_path = os.path.join(output_dir, "translated_transcription.json")
    with open(transcription_path, 'w', encoding='utf-8') as f:
        json.dump(translations, f, indent=2, ensure_ascii=False)
    
    print(f"\nAudio generation completed!")
    print(f"Successful generations: {len(generated_segments)}/{len(translations['segments'])}")
    print(f"Skipped segments (< 0.5s): {len(skipped_segments)}")
    print(f"Report saved to: {report_path}")
    
    # Clear GPU cache
    if available_gpus:
        clear_gpu_cache(gpu_id)
    
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
    parser.add_argument("--gpu-id", type=int, default=None,
                       help="Specific GPU ID to use (default: auto-select)")
    parser.add_argument("--parallel", action="store_true",
                       help="Use parallel processing across multiple GPUs")
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
        args.confidence_threshold,
        args.gpu_id,
        args.parallel
    )
    
    if not success:
        print("Audio generation failed!")
        sys.exit(1)
    else:
        print("Audio generation completed successfully!")


if __name__ == "__main__":
    main() 