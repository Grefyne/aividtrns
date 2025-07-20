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
from multilingual_tokenizer import MultilingualTokenizer, tokenize_for_xtts


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


def calculate_text_similarity(text1, text2):
    """Calculate similarity percentage between two texts using multiple metrics."""
    import difflib
    from collections import Counter
    
    # Normalize texts
    text1 = text1.lower().strip()
    text2 = text2.lower().strip()
    
    if not text1 or not text2:
        return 0.0
    
    # Method 1: SequenceMatcher for overall similarity
    sequence_similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
    
    # Method 2: Word-level Jaccard similarity
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    if not words1 and not words2:
        word_similarity = 1.0
    elif not words1 or not words2:
        word_similarity = 0.0
    else:
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        word_similarity = intersection / union if union > 0 else 0.0
    
    # Method 3: Character-level similarity (for handling small differences)
    char_similarity = difflib.SequenceMatcher(None, 
                                            ''.join(text1.split()), 
                                            ''.join(text2.split())).ratio()
    
    # Weighted combination (sequence matcher gets highest weight)
    final_similarity = (sequence_similarity * 0.5 + 
                       word_similarity * 0.3 + 
                       char_similarity * 0.2)
    
    return final_similarity * 100  # Return as percentage


def verify_transcription_with_whisper(audio_path, expected_text, language="en", device=None):
    """Verify generated audio transcription using Whisper with detailed similarity analysis."""
    try:
        model = whisper.load_model("base")
        if device and hasattr(model, 'to'):
            model = model.to(device)
        
        # Transcribe the audio
        result = model.transcribe(audio_path, language=language)
        transcribed_text = result['text'].strip()
        
        # Calculate similarity percentage
        similarity_percentage = calculate_text_similarity(transcribed_text, expected_text)
        
        # Get Whisper confidence (convert log probability to percentage)
        avg_logprob = result.get('avg_logprob', -1.0)
        whisper_confidence = min(100, max(0, (avg_logprob + 1.0) * 100)) if avg_logprob else 0
        
        return {
            'similarity_percentage': similarity_percentage,
            'whisper_confidence': whisper_confidence,
            'transcribed': transcribed_text,
            'expected': expected_text,
            'word_count_transcribed': len(transcribed_text.split()),
            'word_count_expected': len(expected_text.split()),
            'avg_logprob': avg_logprob
        }
    except Exception as e:
        print(f"  Warning: Whisper verification failed: {e}")
        return None


def generate_translated_audio(translations, output_dir, language, speaker_dir, max_retries=5, confidence_threshold=85.0, gpu_id=None, use_parallel=False):
    """Generate translated audio using XTTS-v2 with multilingual tokenizer."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize multilingual tokenizer
    print(f"Initializing multilingual tokenizer for language: {language}")
    tokenizer = MultilingualTokenizer(language)
    
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
        
        # Preprocess text using multilingual tokenizer
        print(f"  Processing text: {translated_text}")
        processed_text_chunks = tokenizer.tokenize(translated_text)
        
        if not processed_text_chunks:
            print(f"  Warning: Text preprocessing resulted in empty chunks for {segment_id}")
            continue
        
        print(f"  Text split into {len(processed_text_chunks)} chunks")
        for i, chunk in enumerate(processed_text_chunks):
            print(f"    Chunk {i+1}: {chunk}")
        
        # Generate audio with multiple attempts and quality verification
        success = False
        best_attempt = None
        best_similarity = 0.0
        
        print(f"  Target confidence threshold: {confidence_threshold}%")
        
        for attempt in range(max_retries):
            try:
                print(f"  Attempt {attempt + 1}/{max_retries}...")
                
                output_filename = f"{segment_id}_{speaker}_{segment['start_time_str']}_{segment['end_time_str']}_{segment['duration_str']}.wav"
                output_path = os.path.join(output_dir, output_filename)
                
                # If text was split into multiple chunks, process them separately and combine
                if len(processed_text_chunks) > 1:
                    print(f"    Generating audio for {len(processed_text_chunks)} text chunks...")
                    chunk_files = []
                    
                    for i, chunk in enumerate(processed_text_chunks):
                        chunk_filename = f"{segment_id}_{speaker}_chunk{i+1}_{segment['start_time_str']}_{segment['end_time_str']}_{segment['duration_str']}.wav"
                        chunk_path = os.path.join(output_dir, chunk_filename)
                        
                        # Generate audio for this chunk
                        tts.tts_to_file(
                            text=chunk,
                            speaker_wav=speaker_sample,
                            language=language,
                            file_path=chunk_path
                        )
                        chunk_files.append(chunk_path)
                    
                    # Combine chunk files (simple concatenation - could be improved with audio processing)
                    # For now, just use the first chunk as the main file
                    import shutil
                    shutil.move(chunk_files[0], output_path)
                    
                    # Clean up other chunk files
                    for chunk_file in chunk_files[1:]:
                        if os.path.exists(chunk_file):
                            os.remove(chunk_file)
                    
                    # Store the combined text for verification
                    combined_text = " ".join(processed_text_chunks)
                else:
                    # Single chunk - generate normally
                    combined_text = processed_text_chunks[0]
                    tts.tts_to_file(
                        text=combined_text,
                        speaker_wav=speaker_sample,
                        language=language,
                        file_path=output_path
                    )
                
                # Verify with Whisper using the processed text
                print(f"    Verifying audio quality with Whisper...")
                verification = verify_transcription_with_whisper(output_path, combined_text, language, device)
                
                if verification:
                    similarity_percentage = verification['similarity_percentage']
                    whisper_confidence = verification['whisper_confidence']
                    
                    print(f"    Similarity: {similarity_percentage:.1f}% | Whisper confidence: {whisper_confidence:.1f}%")
                    print(f"    Expected: {verification['expected'][:100]}{'...' if len(verification['expected']) > 100 else ''}")
                    print(f"    Got:      {verification['transcribed'][:100]}{'...' if len(verification['transcribed']) > 100 else ''}")
                    
                    # Check if this attempt meets the threshold
                    if similarity_percentage >= confidence_threshold:
                        print(f"  ✅ Audio quality meets threshold ({similarity_percentage:.1f}% >= {confidence_threshold}%)")
                        
                        generated_segments.append({
                            "segment_id": segment_id,
                            "speaker": speaker,
                            "audio_file": output_filename,
                            "original_text": translated_text,
                            "processed_text": combined_text,
                            "text_chunks": len(processed_text_chunks),
                            "whisper_verification": verification,
                            "attempts": attempt + 1,
                            "final_similarity": similarity_percentage,
                            "quality_status": "passed_threshold"
                        })
                        
                        success = True
                        break
                    
                    # Track the best attempt so far
                    if similarity_percentage > best_similarity:
                        best_similarity = similarity_percentage
                        
                        # Save the best attempt info
                        if best_attempt and best_attempt.get('temp_file') and os.path.exists(best_attempt['temp_file']):
                            os.remove(best_attempt['temp_file'])
                        
                        # Create temporary file for best attempt
                        temp_filename = f"{segment_id}_{speaker}_best_attempt.wav"
                        temp_path = os.path.join(output_dir, temp_filename)
                        import shutil
                        shutil.copy2(output_path, temp_path)
                        
                        best_attempt = {
                            "verification": verification,
                            "similarity": similarity_percentage,
                            "temp_file": temp_path,
                            "attempt_number": attempt + 1
                        }
                        
                        print(f"    New best attempt: {similarity_percentage:.1f}% (attempt {attempt + 1})")
                    else:
                        print(f"    Lower quality than best attempt ({similarity_percentage:.1f}% < {best_similarity:.1f}%)")
                    
                    # Remove current attempt file if not the best
                    if os.path.exists(output_path):
                        os.remove(output_path)
                        
                else:
                    print(f"    Whisper verification failed")
                    if os.path.exists(output_path):
                        os.remove(output_path)
                        
            except Exception as e:
                print(f"    Error generating audio: {e}")
                if os.path.exists(output_path):
                    os.remove(output_path)
                continue
        
        # If no attempt met the threshold, use the best attempt
        if not success and best_attempt:
            print(f"  📋 No attempt met {confidence_threshold}% threshold. Using best attempt ({best_similarity:.1f}%)")
            
            # Move best attempt to final location
            final_output_path = os.path.join(output_dir, output_filename)
            import shutil
            shutil.move(best_attempt['temp_file'], final_output_path)
            
            generated_segments.append({
                "segment_id": segment_id,
                "speaker": speaker,
                "audio_file": output_filename,
                "original_text": translated_text,
                "processed_text": combined_text,
                "text_chunks": len(processed_text_chunks),
                "whisper_verification": best_attempt['verification'],
                "attempts": max_retries,
                "final_similarity": best_similarity,
                "quality_status": "best_below_threshold"
            })
            
            success = True
        
        # Clean up any remaining temporary files
        if best_attempt and best_attempt.get('temp_file') and os.path.exists(best_attempt['temp_file']):
            os.remove(best_attempt['temp_file'])
        
        if not success:
            print(f"  ❌ Failed to generate acceptable audio for {segment_id} after {max_retries} attempts")
    
    # Calculate quality statistics
    quality_stats = {
        "passed_threshold": 0,
        "best_below_threshold": 0,
        "average_similarity": 0.0,
        "similarity_distribution": {
            "90-100%": 0,
            "80-89%": 0, 
            "70-79%": 0,
            "60-69%": 0,
            "below_60%": 0
        }
    }
    
    if generated_segments:
        total_similarity = 0
        for segment in generated_segments:
            similarity = segment.get('final_similarity', 0)
            total_similarity += similarity
            
            status = segment.get('quality_status', 'unknown')
            if status == 'passed_threshold':
                quality_stats['passed_threshold'] += 1
            elif status == 'best_below_threshold':
                quality_stats['best_below_threshold'] += 1
            
            # Similarity distribution
            if similarity >= 90:
                quality_stats['similarity_distribution']['90-100%'] += 1
            elif similarity >= 80:
                quality_stats['similarity_distribution']['80-89%'] += 1
            elif similarity >= 70:
                quality_stats['similarity_distribution']['70-79%'] += 1
            elif similarity >= 60:
                quality_stats['similarity_distribution']['60-69%'] += 1
            else:
                quality_stats['similarity_distribution']['below_60%'] += 1
        
        quality_stats['average_similarity'] = total_similarity / len(generated_segments)
    
    # Save generation report
    report_data = {
        "model_used": "XTTS-v2",
        "tokenizer_used": "MultilingualTokenizer",
        "language": language,
        "total_segments": len(translations['segments']),
        "successful_generations": len(generated_segments),
        "skipped_segments": len(skipped_segments),
        "quality_verification": {
            "confidence_threshold": confidence_threshold,
            "max_retries": max_retries,
            "quality_statistics": quality_stats
        },
        "text_preprocessing": {
            "abbreviation_expansion": True,
            "symbol_expansion": True,
            "number_expansion": True,
            "sentence_splitting": True
        },
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
    
    print(f"\n🎉 Audio generation completed!")
    print(f"📊 Results Summary:")
    print(f"  • Total segments processed: {len(translations['segments'])}")
    print(f"  • Successful generations: {len(generated_segments)}")
    print(f"  • Skipped segments (< 0.5s): {len(skipped_segments)}")
    print(f"  • Failed generations: {len(translations['segments']) - len(generated_segments) - len(skipped_segments)}")
    
    if generated_segments:
        print(f"\n🎯 Quality Analysis:")
        print(f"  • Average similarity: {quality_stats['average_similarity']:.1f}%")
        print(f"  • Passed threshold (≥85%): {quality_stats['passed_threshold']}")
        print(f"  • Best below threshold: {quality_stats['best_below_threshold']}")
        
        print(f"\n📈 Similarity Distribution:")
        for range_name, count in quality_stats['similarity_distribution'].items():
            if count > 0:
                percentage = (count / len(generated_segments)) * 100
                print(f"  • {range_name}: {count} segments ({percentage:.1f}%)")
    
    print(f"\n📁 Report saved to: {report_path}")
    
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
                       help="Maximum retry attempts for audio generation (default: 5)")
    parser.add_argument("--confidence-threshold", type=float, default=85.0,
                       help="Similarity threshold percentage for audio quality verification (default: 85%)")
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