#!/usr/bin/env python3
"""
Enhanced Audio Generation Script using XTTS-v2 with Multilingual Tokenizer
Includes Whisper verification, multi-attempt generation, and maximum quality settings.
"""

import os
import sys
import json
import argparse
import whisper
import torch
import subprocess
import re
from TTS.api import TTS
from pathlib import Path
from gpu_utils import get_device, setup_multi_gpu_processing, clear_gpu_cache, parallel_process_with_gpus
from multilingual_tokenizer import VoiceBpeTokenizer, split_sentence, multilingual_cleaners

# Fix PyTorch 2.6+ weights loading issue
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    torch.serialization.add_safe_globals([XttsConfig])
except Exception:
    pass  # Fallback to older PyTorch behavior


def get_language_char_limit(language):
    """
    Get optimal character limits per language to prevent truncation.
    Uses the same limits as the multilingual tokenizer but more conservative.
    """
    # Use the tokenizer's char limits but make them more conservative for anti-truncation
    tokenizer = VoiceBpeTokenizer()
    base_limit = tokenizer.char_limits.get(language, 150)
    
    # Make limits more conservative to prevent truncation
    conservative_limits = {
        'es': min(base_limit, 180),  # Spanish
        'en': min(base_limit, 200),  # English  
        'fr': min(base_limit, 170),  # French
        'de': min(base_limit, 160),  # German
        'it': min(base_limit, 180),  # Italian
        'pt': min(base_limit, 180),  # Portuguese
        'ru': min(base_limit, 150),  # Russian
        'zh': min(base_limit, 120),  # Chinese
        'ja': min(base_limit, 120),  # Japanese
        'ko': min(base_limit, 120),  # Korean
        'ar': min(base_limit, 140),  # Arabic
        'hi': min(base_limit, 140),  # Hindi
        'tr': min(base_limit, 160),  # Turkish
        'pl': min(base_limit, 160),  # Polish
        'cs': min(base_limit, 160),  # Czech
        'nl': min(base_limit, 170),  # Dutch
        'hu': min(base_limit, 150),  # Hungarian
        'ur': min(base_limit, 150),  # Urdu
        'sd': min(base_limit, 180),  # Sindhi
    }
    
    return conservative_limits.get(language, min(base_limit, 150))


def preprocess_text_enhanced(text, language):
    """
    Enhanced text preprocessing using the multilingual tokenizer.
    """
    if not text or not text.strip():
        return ""
    
    # Use the comprehensive multilingual cleaner
    cleaned_text = multilingual_cleaners(text, language)
    
    # Additional cleanup for TTS stability
    # Remove excessive punctuation that can cause issues
    cleaned_text = re.sub(r'\.{3,}', '...', cleaned_text)  # Normalize ellipsis
    cleaned_text = re.sub(r'[!]{2,}', '!', cleaned_text)   # Reduce multiple exclamations
    cleaned_text = re.sub(r'[?]{2,}', '?', cleaned_text)   # Reduce multiple questions
    
    # Ensure proper sentence endings for better TTS flow
    if cleaned_text and not cleaned_text[-1] in '.!?':
        cleaned_text += '.'
    
    return cleaned_text.strip()


def split_text_smart(text, language, max_length=None):
    """
    Smart text splitting using the multilingual tokenizer's split_sentence function
    with enhanced anti-truncation logic.
    """
    if not text or not text.strip():
        return []
    
    if max_length is None:
        max_length = get_language_char_limit(language)
    
    # If text is short enough, return as single chunk
    if len(text) <= max_length:
        return [text.strip()]
    
    # Use the multilingual tokenizer's smart splitting (already imported)
    
    try:
        # Use the tokenizer's advanced sentence splitting
        chunks = split_sentence(text, language, max_length)
        
        # Ensure all chunks are within limits
        final_chunks = []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
                
            if len(chunk) <= max_length:
                final_chunks.append(chunk)
            else:
                # Further split oversized chunks by words
                words = chunk.split()
                current_chunk = ""
                
                for word in words:
                    test_chunk = current_chunk + (" " if current_chunk else "") + word
                    
                    if len(test_chunk) <= max_length:
                        current_chunk = test_chunk
                    else:
                        if current_chunk:
                            final_chunks.append(current_chunk.strip())
                        current_chunk = word
                
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
        
        return [chunk for chunk in final_chunks if chunk.strip()]
        
    except Exception as e:
        print(f"  ⚠️  Fallback to simple word-based splitting due to: {e}")
        # Fallback to simple word-based splitting
        words = text.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            test_chunk = current_chunk + (" " if current_chunk else "") + word
            
            if len(test_chunk) <= max_length:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = word
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


def calculate_similarity(text1, text2):
    """
    Calculate similarity between two texts using multiple metrics.
    """
    if not text1 or not text2:
        return 0.0
    
    # Normalize texts for comparison
    def normalize_text(text):
        # Convert to lowercase and remove extra whitespace
        text = re.sub(r'\s+', ' ', text.lower().strip())
        # Remove punctuation for comparison
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    norm_text1 = normalize_text(text1)
    norm_text2 = normalize_text(text2)
    
    if not norm_text1 or not norm_text2:
        return 0.0
    
    # Sequence matcher for overall similarity
    from difflib import SequenceMatcher
    sequence_similarity = SequenceMatcher(None, norm_text1, norm_text2).ratio()
    
    # Word-level Jaccard similarity
    words1 = set(norm_text1.split())
    words2 = set(norm_text2.split())
    if words1 or words2:
        word_similarity = len(words1.intersection(words2)) / len(words1.union(words2))
    else:
        word_similarity = 0.0
    
    # Character-level similarity
    if len(norm_text1) > 0 and len(norm_text2) > 0:
        char_similarity = SequenceMatcher(None, norm_text1, norm_text2).ratio()
    else:
        char_similarity = 0.0
    
    # Weighted combination
    final_similarity = (
        sequence_similarity * 0.5 +
        word_similarity * 0.3 +
        char_similarity * 0.2
    )
    
    return final_similarity * 100  # Return as percentage


def load_translations(input_file):
    """Load translated transcriptions from JSON file."""
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return None
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data['segments'])} segments from {input_file}")
        return data
    except Exception as e:
        print(f"Error loading translations: {e}")
        return None


def find_speaker_samples(speaker_dir):
    """Find available speaker sample files."""
    if not os.path.exists(speaker_dir):
        print(f"Warning: Speaker directory not found: {speaker_dir}")
        return {}
    
    speaker_samples = {}
    for file in os.listdir(speaker_dir):
        if file.endswith(('.wav', '.mp3', '.flac')):
            # Extract speaker ID from filename
            if 'speaker00' in file.lower() or 'speaker_00' in file.lower():
                speaker_samples['SPEAKER_00'] = os.path.join(speaker_dir, file)
            elif 'speaker01' in file.lower() or 'speaker_01' in file.lower():
                speaker_samples['SPEAKER_01'] = os.path.join(speaker_dir, file)
    
    print(f"Found {len(speaker_samples)} speaker samples")
    return speaker_samples


def verify_transcription_with_whisper(audio_path, expected_text, language="en", device=None):
    """Verify generated audio quality using Whisper transcription."""
    try:
        # Load Whisper model (use small model for speed)
        model = whisper.load_model("small", device=device)
        
        # Transcribe the generated audio
        result = model.transcribe(audio_path, language=language)
        transcribed_text = result["text"].strip()
        
        # Calculate similarity percentage
        similarity_percentage = calculate_similarity(transcribed_text, expected_text)
        
        # Get Whisper confidence (segments have confidence scores)
        whisper_confidence = 0.0
        if "segments" in result and result["segments"]:
            # Average confidence of all segments
            confidences = [seg.get("avg_logprob", 0.0) for seg in result["segments"]]
            if confidences:
                avg_log_prob = sum(confidences) / len(confidences)
                # Convert log probability to percentage (rough approximation)
                whisper_confidence = max(0.0, min(100.0, (avg_log_prob + 1.0) * 100))
        
        return {
            "similarity": similarity_percentage,
            "whisper_confidence": whisper_confidence,
            "transcribed_text": transcribed_text,
            "expected_text": expected_text
        }
        
    except Exception as e:
        print(f"    ⚠️  Whisper verification failed: {e}")
        return {
            "similarity": 0.0,
            "whisper_confidence": 0.0,
            "transcribed_text": "",
            "expected_text": expected_text
        }


def generate_audio_segment(tts, segment, speaker_sample, language, output_dir, 
                          max_retries=5, confidence_threshold=85.0):
    """
    Generate audio for a single segment with quality verification and retry logic.
    """
    segment_id = segment['segment_id']
    speaker = segment['speaker']
    translated_text = segment['translated_transcription']
    
    print(f"  🎵 Generating audio for segment {segment_id}...")
    print(f"  👤 Speaker: {speaker}")
    print(f"  📝 Text: {translated_text}")
    
    # Enhanced text preprocessing
    cleaned_text = preprocess_text_enhanced(translated_text, language)
    print(f"  ✨ Cleaned: {cleaned_text}")
    
    # Get language-specific character limit
    char_limit = get_language_char_limit(language)
    print(f"  📏 Char limit for {language}: {char_limit}")
    
    # Smart text splitting with anti-truncation
    if len(cleaned_text) > char_limit:
        print(f"  ✂️  Text too long ({len(cleaned_text)} chars), using smart splitting...")
        text_chunks = split_text_smart(cleaned_text, language, char_limit)
        print(f"  📦 Split into {len(text_chunks)} chunks")
    else:
        text_chunks = [cleaned_text]
        print(f"  ✅ Text fits in single chunk")
    
    if not text_chunks:
        print(f"  ❌ No valid text chunks generated")
        return None
    
    print(f"  🎯 Text chunks:")
    for i, chunk in enumerate(text_chunks):
        print(f"    Chunk {i+1}: ({len(chunk)} chars) {chunk[:60]}...")
    
    # Generate filename
    duration_str = f"{segment['duration']:.3f}".replace('.', '-')
    output_filename = f"{segment_id}_{speaker}_{segment['start_time_str']}_{segment['end_time_str']}_{duration_str}.wav"
    output_path = os.path.join(output_dir, output_filename)
    
    best_similarity = 0.0
    best_audio_path = None
    best_verification = None
    
    print(f"  🎯 Target confidence threshold: {confidence_threshold}%")
    
    for attempt in range(max_retries):
        print(f"  🎬 Attempt {attempt + 1}/{max_retries}...")
        
        try:
            # Generate audio with maximum quality settings
            if len(text_chunks) > 1:
                # Multiple chunks - generate separately and concatenate
                print(f"    🔗 Generating {len(text_chunks)} chunks...")
                chunk_files = []
                
                for i, chunk in enumerate(text_chunks):
                    chunk_filename = f"{segment_id}_{speaker}_chunk{i+1}_{segment['start_time_str']}_{segment['end_time_str']}_{duration_str}.wav"
                    chunk_path = os.path.join(output_dir, chunk_filename)
                    
                    # Generate with maximum quality settings
                    tts.tts_to_file(
                        text=chunk,
                        speaker_wav=speaker_sample,
                        language=language,
                        file_path=chunk_path,
                        temperature=0.65,           # Stability
                        repetition_penalty=5.0,    # Reduce artifacts
                        top_k=20,                  # Higher quality
                        top_p=0.75,                # Predictable output
                        split_sentences=True       # Enhanced processing
                    )
                    chunk_files.append(chunk_path)
                
                # Concatenate chunks using FFmpeg
                print(f"    🔗 Concatenating {len(chunk_files)} chunks...")
                concat_cmd = [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", "/dev/stdin", "-c", "copy", output_path
                ]
                
                # Create input list for FFmpeg
                file_list = "\n".join([f"file '{os.path.abspath(f)}'" for f in chunk_files])
                
                result = subprocess.run(concat_cmd, input=file_list, text=True, 
                                      capture_output=True, check=True)
                
                # Clean up chunk files
                for chunk_file in chunk_files:
                    if os.path.exists(chunk_file):
                        os.remove(chunk_file)
                
                combined_text = " ".join(text_chunks)
            else:
                # Single chunk - generate directly
                combined_text = text_chunks[0]
                tts.tts_to_file(
                    text=combined_text,
                    speaker_wav=speaker_sample,
                    language=language,
                    file_path=output_path,
                    temperature=0.65,           # Stability
                    repetition_penalty=5.0,    # Reduce artifacts
                    top_k=20,                  # Higher quality
                    top_p=0.75,                # Predictable output
                    split_sentences=True       # Enhanced processing
                )
            
            # Get actual audio duration
            try:
                duration_cmd = [
                    "ffprobe", "-v", "quiet", "-show_entries", "format=duration", 
                    "-of", "csv=p=0", output_path
                ]
                duration_result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
                actual_audio_duration = float(duration_result.stdout.strip())
            except:
                actual_audio_duration = None
            
            print(f"    ✅ Audio generated successfully")
            if actual_audio_duration:
                print(f"    ⏱️  Duration: {actual_audio_duration:.3f}s")
            
            # Verify quality with Whisper
            print(f"    🎤 Verifying quality with Whisper...")
            verification = verify_transcription_with_whisper(output_path, combined_text, language)
            
            similarity = verification["similarity"]
            whisper_conf = verification["whisper_confidence"]
            
            print(f"    📊 Similarity: {similarity:.1f}% | Whisper confidence: {whisper_conf:.1f}%")
            print(f"    📝 Expected: {combined_text[:60]}...")
            print(f"    🎤 Got:      {verification['transcribed_text'][:60]}...")
            
            if similarity >= confidence_threshold:
                print(f"    ✅ Quality meets threshold ({similarity:.1f}% >= {confidence_threshold}%)")
                return {
                    "success": True,
                    "output_path": output_path,
                    "similarity": similarity,
                    "whisper_confidence": whisper_conf,
                    "attempts": attempt + 1,
                    "text_chunks": len(text_chunks),
                    "combined_text": combined_text,
                    "actual_duration": actual_audio_duration,
                    "quality_status": "passed_threshold"
                }
            else:
                if similarity > best_similarity:
                    print(f"    📈 New best attempt: {similarity:.1f}% (attempt {attempt + 1})")
                    best_similarity = similarity
                    best_audio_path = output_path + f"_best_attempt_{attempt + 1}.wav"
                    # Save this attempt as the best
                    if os.path.exists(output_path):
                        import shutil
                        shutil.copy2(output_path, best_audio_path)
                    best_verification = verification.copy()
                    best_verification["actual_duration"] = actual_audio_duration
                else:
                    print(f"    📉 Lower quality than best attempt ({similarity:.1f}% < {best_similarity:.1f}%)")
                
                # Clean up this attempt if it's not the best
                if similarity < best_similarity and os.path.exists(output_path):
                    os.remove(output_path)
            
        except Exception as e:
            print(f"    ❌ Generation failed: {e}")
            continue
    
    # Use best attempt if no attempt met threshold
    if best_audio_path and os.path.exists(best_audio_path):
        # Move best attempt to final location
        import shutil
        shutil.move(best_audio_path, output_path)
        
        print(f"  📋 No attempt met {confidence_threshold}% threshold. Using best attempt ({best_similarity:.1f}%)")
        
        return {
            "success": True,
            "output_path": output_path,
            "similarity": best_similarity,
            "whisper_confidence": best_verification.get("whisper_confidence", 0.0),
            "attempts": max_retries,
            "text_chunks": len(text_chunks),
            "combined_text": best_verification.get("expected_text", ""),
            "actual_duration": best_verification.get("actual_duration"),
            "quality_status": "best_below_threshold"
        }
    
    print(f"  ❌ All {max_retries} attempts failed")
    return None


def main():
    parser = argparse.ArgumentParser(description='Enhanced audio generation with multilingual tokenizer')
    parser.add_argument('--input', required=True, help='Input JSON file with translations')
    parser.add_argument('--output', required=True, help='Output directory for audio files')
    parser.add_argument('--language', required=True, help='Target language code (e.g., es, fr, de)')
    parser.add_argument('--speaker-dir', required=True, help='Directory containing speaker samples')
    parser.add_argument('--max-retries', type=int, default=5, help='Maximum retry attempts per segment')
    parser.add_argument('--confidence-threshold', type=float, default=85.0, help='Quality threshold percentage')
    parser.add_argument('--gpu-id', type=int, help='Specific GPU ID to use')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing (experimental)')
    
    args = parser.parse_args()
    
    # Initialize multilingual tokenizer
    print(f"🧠 Initializing multilingual tokenizer for language: {args.language}")
    
    # Load translations
    translations = load_translations(args.input)
    if not translations:
        return False
    
    # Find speaker samples
    speaker_samples = find_speaker_samples(args.speaker_dir)
    if not speaker_samples:
        print("Error: No speaker samples found")
        return False
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Setup GPU
    if args.parallel:
        device = setup_multi_gpu_processing()
        print("Using parallel GPU processing")
    else:
        device = get_device(args.gpu_id)
        print(f"Using single GPU: {device}")
    
    # Initialize XTTS model with maximum quality settings
    print("🎵 Loading XTTS-v2 model with maximum quality settings...")
    try:
        # Set weights_only=False for PyTorch 2.6+ compatibility
        original_load = torch.load
        torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False) if 'weights_only' not in kwargs else original_load(*args, **kwargs)
        
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        
        # Restore original torch.load
        torch.load = original_load
        
        if hasattr(tts, 'to') and device.type == 'cuda':
            tts = tts.to(device)
        
        print("  ✅ TTS model loaded successfully with maximum quality settings:")
        print("  • Temperature: 0.65 (stability)")
        print("  • Repetition penalty: 5.0 (reduce repetition)")
        print("  • Top-k: 20 (higher quality)")
        print("  • Top-p: 0.75 (more predictable)")
        print("  • Split sentences: enabled (better quality)")
        print("  • Enhanced multilingual tokenizer: enabled")
        print("  • Advanced anti-truncation: enabled")
        
    except Exception as e:
        print(f"Error loading XTTS model: {e}")
        return False
    
    print(f"🎯 Processing {len(translations['segments'])} segments...")
    
    # Process segments
    results = []
    successful = 0
    failed = 0
    skipped = 0
    
    for i, segment in enumerate(translations['segments']):
        print(f"\n{'='*80}")
        print(f"Processing segment {i+1}/{len(translations['segments'])}: {segment['segment_id']}")
        
        # Skip very short segments
        if segment['duration'] < 0.5:
            print(f"  ⏭️  Skipping short segment ({segment['duration']:.3f}s)")
            skipped += 1
            continue
        
        # Get speaker sample
        speaker = segment['speaker']
        if speaker not in speaker_samples:
            print(f"  ❌ No speaker sample found for {speaker}")
            failed += 1
            continue
        
        speaker_sample = speaker_samples[speaker]
        
        # Generate audio
        result = generate_audio_segment(
            tts, segment, speaker_sample, args.language, args.output,
            args.max_retries, args.confidence_threshold
        )
        
        if result:
            result['segment_id'] = segment['segment_id']
            result['speaker'] = speaker
            result['expected_duration'] = segment['duration']
            results.append(result)
            successful += 1
            print(f"  ✅ Success: {result['similarity']:.1f}% similarity")
        else:
            failed += 1
            print(f"  ❌ Failed after {args.max_retries} attempts")
    
    # Generate summary report
    print(f"\n🎉 Audio generation completed!")
    print(f"📊 Results Summary:")
    print(f"  • Total segments processed: {len(translations['segments'])}")
    print(f"  • Successful generations: {successful}")
    print(f"  • Failed generations: {failed}")
    print(f"  • Skipped segments (< 0.5s): {skipped}")
    
    if results:
        # Quality analysis
        similarities = [r['similarity'] for r in results]
        avg_similarity = sum(similarities) / len(similarities)
        passed_threshold = len([r for r in results if r['quality_status'] == 'passed_threshold'])
        
        print(f"\n🎯 Quality Analysis:")
        print(f"  • Average similarity: {avg_similarity:.1f}%")
        print(f"  • Passed threshold (≥{args.confidence_threshold}%): {passed_threshold}")
        print(f"  • Best below threshold: {len(results) - passed_threshold}")
        
        # Duration analysis
        expected_durations = [r['expected_duration'] for r in results if r.get('expected_duration')]
        actual_durations = [r['actual_duration'] for r in results if r.get('actual_duration')]
        
        if expected_durations and actual_durations:
            total_expected = sum(expected_durations)
            total_actual = sum(actual_durations)
            print(f"\n📊 Duration Analysis:")
            print(f"  • Total expected duration: {total_expected:.3f}s")
            print(f"  • Total actual duration: {total_actual:.3f}s")
            print(f"  • Duration ratio: {total_actual/total_expected:.3f}")
        
        # Save detailed report
        report = {
            "generation_info": {
                "language": args.language,
                "total_segments": len(translations['segments']),
                "successful": successful,
                "failed": failed,
                "skipped": skipped,
                "max_retries": args.max_retries,
                "confidence_threshold": args.confidence_threshold,
                "average_similarity": avg_similarity,
                "passed_threshold": passed_threshold
            },
            "segments": results
        }
        
        report_path = os.path.join(args.output, "generation_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Report saved to: {report_path}")
    
    # Clear GPU cache
    clear_gpu_cache(device)
    print("🧹 GPU cache cleared")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 