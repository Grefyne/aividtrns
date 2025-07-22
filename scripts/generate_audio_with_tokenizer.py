#!/usr/bin/env python3
"""
Enhanced Audio Generation Script using XTTS-v2 with Multilingual Tokenizer
Includes Whisper verification, multi-attempt generation, and maximum quality settings.

DEBUG VERSION: Only processes segment 9 and preserves all chunk files for debugging truncation issues.
"""
# Set Hugging Face cache location to suppress warnings and improve performance
# Set TensorFlow oneDNN options to 0 for consistent numerical results
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings

warnings.filterwarnings(
    "ignore", 
    message=".*GPT2InferenceModel has generative capabilities.*"
)

warnings.filterwarnings(
    "ignore", 
    message="The attention mask.*"
)

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
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

torch.backends.cudnn.allow_tf32 = True          # keep TF32 if you like
torch.backends.cudnn.benchmark = False          # prevents re-planning every shape

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
    Now aggressively normalizes punctuation to reduce unnecessary pauses.
    """
    import re
    if not text or not text.strip():
        return ""
    
    # Use the comprehensive multilingual cleaner
    cleaned_text = multilingual_cleaners(text, language)
    
    # Aggressive punctuation normalization
    # Replace ellipses and multiple periods with a single period
    cleaned_text = re.sub(r'\.{2,}', '.', cleaned_text)
    # Remove multiple exclamation/question marks
    cleaned_text = re.sub(r'[!]{2,}', '!', cleaned_text)
    cleaned_text = re.sub(r'[?]{2,}', '?', cleaned_text)
    # Remove all commas (optional, can be adjusted)
    cleaned_text = cleaned_text.replace(',', '')
    # Remove any stray semicolons
    cleaned_text = cleaned_text.replace(';', '')
    # Remove double spaces
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)
    # Ensure proper sentence endings for better TTS flow
    if cleaned_text and not cleaned_text[-1] in '.!?':
        cleaned_text += '.'
    
    return cleaned_text.strip()


def smart_chunk_text(text, language, char_limit):
    """
    Split text into chunks that are safe for the tokenizer and TTS.
    1. Split by sentences (using NLTK for English, fallback to period for others).
    2. If a sentence is too long, try to split at commas or other punctuation (using the original sentence).
    3. If still too long, split at the nearest space before the char limit.
    4. Only then apply aggressive cleaning/tokenization to each chunk.
    """
    # Step 1: Sentence splitting
    if language == 'en':
        sentences = sent_tokenize(text)
    else:
        # Fallback: split on period
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    chunks = []
    for sent in sentences:
        # Step 2: If sentence is too long, try to split at commas or other punctuation
        if len(sent) > char_limit - 10:
            # Try to split at commas, semicolons, or dashes
            split_points = [m.start() for m in re.finditer(r'[,:;\-]', sent)]
            last = 0
            for idx in split_points:
                if idx - last > char_limit - 10:
                    # If the chunk is still too long, split at the last space before char_limit
                    sub = sent[last:idx].strip()
                    if len(sub) > char_limit - 10:
                        # Step 3: Split at nearest space before char_limit
                        safe_idx = sub.rfind(' ', 0, char_limit - 10)
                        if safe_idx > 0:
                            chunks.append(sub[:safe_idx].strip())
                            last = last + safe_idx
                        else:
                            chunks.append(sub)
                            last = idx
                    else:
                        chunks.append(sub)
                        last = idx
            # Add the last chunk
            if last < len(sent):
                sub = sent[last:].strip()
                if len(sub) > char_limit - 10:
                    safe_idx = sub.rfind(' ', 0, char_limit - 10)
                    if safe_idx > 0:
                        chunks.append(sub[:safe_idx].strip())
                        chunks.append(sub[safe_idx:].strip())
                    else:
                        chunks.append(sub)
                else:
                    chunks.append(sub)
        else:
            chunks.append(sent.strip())
    # Step 4: Clean/tokenize each chunk
    cleaned_chunks = [preprocess_text_enhanced(chunk, language) for chunk in chunks if chunk.strip()]
    return [c for c in cleaned_chunks if c]


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


# Add a helper for colored output
RED = '\033[91m'
BOLD = '\033[1m'
RESET = '\033[0m'

def printq(*args, **kwargs):
    """Print only if not in quiet mode."""
    if not getattr(printq, 'quiet', False):
        print(*args, **kwargs)


def verify_transcription_with_whisper(audio_path, expected_text, language="en", device=None, quiet=False, confidence_threshold=85.0):
    """Verify generated audio quality using Whisper transcription."""
    try:
        # Load Whisper model (use small model for speed)
        model = whisper.load_model("small", device=device)
        # Transcribe the generated audio
        result = model.transcribe(audio_path, language=language)
        transcribed_text = result["text"].strip()
        # Normalize both texts using the same preprocessing
        norm_expected = preprocess_text_enhanced(expected_text, language)
        norm_transcribed = preprocess_text_enhanced(transcribed_text, language)
        # Calculate similarity percentage
        similarity_percentage = calculate_similarity(norm_transcribed, norm_expected)
        # Get Whisper confidence (segments have confidence scores)
        whisper_confidence = 0.0
        if "segments" in result and result["segments"]:
            # Average confidence of all segments
            confidences = [seg.get("avg_logprob", 0.0) for seg in result["segments"]]
            if confidences:
                avg_log_prob = sum(confidences) / len(confidences)
                # Convert log probability to percentage (rough approximation)
                whisper_confidence = max(0.0, min(100.0, (avg_log_prob + 1.0) * 100))
        # Only print if not quiet, or always print key info if quiet
        if not quiet:
            print(f"    📝 Full expected (normalized, {len(norm_expected)} chars):\n    {norm_expected}")
            print(f"    🎤 Full got (normalized, {len(norm_transcribed)} chars):\n    {norm_transcribed}")
            print(f"    📝 Full expected (original):\n    {expected_text}")
            print(f"    🎤 Full got (original):\n    {transcribed_text}")
        else:
            # Show in red and bold if similarity below threshold
            highlight = RED + BOLD if similarity_percentage < confidence_threshold else ''
            reset = RESET if similarity_percentage < confidence_threshold else ''
            print(f"    📝 Full expected (normalized, {len(norm_expected)} chars):\n    {highlight}{norm_expected}{reset}")
            print(f"    🎤 Full got (normalized, {len(norm_transcribed)} chars):\n    {highlight}{norm_transcribed}{reset}")
        return {
            "similarity": similarity_percentage,
            "whisper_confidence": whisper_confidence,
            "transcribed_text": norm_transcribed,
            "expected_text": norm_expected
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
                          max_retries=5, confidence_threshold=85.0, quiet=False):
    """
    Generate audio for a single segment with quality verification and retry logic.
    DEBUG: Does NOT delete or modify any chunk files for debugging purposes.
    """
    segment_id = segment['segment_id']
    speaker = segment['speaker']
    # Handle both transcribed and translated formats
    translated_text = segment.get('translated_transcription') or segment.get('transcription', '')
    
    printq(f"  🎵 Generating audio for segment {segment_id}...")
    printq(f"  👤 Speaker: {speaker}")
    printq(f"   Text: {translated_text}")
    
    # Enhanced text preprocessing
    cleaned_text = preprocess_text_enhanced(translated_text, language)
    printq(f"  ✨ Cleaned: {cleaned_text}")
    
    # Get language-specific character limit
    char_limit = get_language_char_limit(language)
    printq(f"  📏 Char limit for {language}: {char_limit}")
    
    # Smart text splitting with anti-truncation
    text_chunks = smart_chunk_text(translated_text, language, char_limit)
    if len(text_chunks) > 1:
        printq(f"  📦 Split into {len(text_chunks)} chunks (smart chunking)")
    else:
        printq(f"  ✅ Text fits in single chunk (smart chunking)")
    
    if not text_chunks:
        printq(f"  ❌ No valid text chunks generated")
        return None
    
    printq(f"  🎯 Text chunks:")
    for i, chunk in enumerate(text_chunks):
        printq(f"    Chunk {i+1}: ({len(chunk)} chars) {chunk[:60]}...")
    
    # Generate filename
    duration_str = f"{segment['duration']:.3f}".replace('.', '-')
    output_filename = f"{segment_id}_{speaker}_{segment['start_time_str']}_{segment['end_time_str']}_{duration_str}.wav"
    output_path = os.path.join(output_dir, output_filename)
    
    best_similarity = 0.0
    best_audio_path = None
    best_verification = None
    
    printq(f"  🎯 Target confidence threshold: {confidence_threshold}%")
    
    # Create unique temporary path for each attempt to avoid conflicts
    temp_dir = os.path.join(output_dir, f"temp_{segment_id}")
    os.makedirs(temp_dir, exist_ok=True)
    
    for attempt in range(max_retries):
        printq(f"  🎬 Attempt {attempt + 1}/{max_retries}...")
        
        # Use temporary path for this attempt
        temp_output_path = os.path.join(temp_dir, f"attempt_{attempt + 1}.wav")
        
        try:
            # Generate audio with maximum quality settings
            if len(text_chunks) > 1:
                # Multiple chunks - generate separately and concatenate
                printq(f"    🔗 Generating {len(text_chunks)} chunks...")
                chunk_files = []
                
                for i, chunk in enumerate(text_chunks):
                    chunk_filename = f"chunk{i+1}_attempt{attempt + 1}.wav"
                    chunk_path = os.path.join(temp_dir, chunk_filename)
                    
                    # Generate with maximum quality settings and natural speed
                    tts.tts_to_file(
                        text=chunk,
                        speaker_wav=speaker_sample,
                        language=language,
                        file_path=chunk_path
                    )
                    chunk_files.append(chunk_path)
                
                # Concatenate chunks using FFmpeg
                printq(f"    🔗 Concatenating {len(chunk_files)} chunks...")
                concat_cmd = [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", "/dev/stdin", "-c", "copy", temp_output_path
                ]
                
                # Create input list for FFmpeg
                file_list = "\n".join([f"file '{os.path.abspath(f)}'" for f in chunk_files])
                
                result = subprocess.run(concat_cmd, input=file_list, text=True, 
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
                
                # After concatenation, delete the chunk files
                for chunk_file in chunk_files:
                    if os.path.exists(chunk_file):
                        os.remove(chunk_file)
                
                # Do NOT delete temp_dir here; wait until all processing is complete
                
                combined_text = " ".join(text_chunks)
            else:
                # Single chunk - generate directly
                combined_text = text_chunks[0]
                tts.tts_to_file(
                    text=combined_text,
                    speaker_wav=speaker_sample,
                    language=language,
                    file_path=temp_output_path
                )
            
            # Get actual audio duration
            try:
                duration_cmd = [
                    "ffprobe", "-v", "quiet", "-show_entries", "format=duration", 
                    "-of", "csv=p=0", temp_output_path
                ]
                duration_result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
                actual_audio_duration = float(duration_result.stdout.strip())
            except:
                actual_audio_duration = None
            
            printq(f"    ✅ Audio generated successfully")
            if actual_audio_duration:
                printq(f"    ⏱️  Duration: {actual_audio_duration:.3f}s")
            
            # Verify quality with Whisper
            if not quiet:
                printq(f"    🎤 Verifying quality with Whisper...")
            else:
                printq(f"    🎤 Verifying quality with Whisper...")
            verification = verify_transcription_with_whisper(temp_output_path, combined_text, language, quiet=quiet, confidence_threshold=confidence_threshold)
            
            similarity = verification["similarity"]
            whisper_conf = verification["whisper_confidence"]
            
            printq(f"    📊 Similarity: {similarity:.1f}% | Whisper confidence: {whisper_conf:.1f}%")
            # Remove any print statements that use [:60] or ... for expected/got
            printq(f"    📝 Expected: {combined_text}")
            printq(f"    🎤 Got:      {verification['transcribed_text']}")
            
            if similarity >= confidence_threshold:
                printq(f"    ✅ Quality meets threshold ({similarity:.1f}% >= {confidence_threshold}%)")
                # Move successful file to final location
                import shutil
                shutil.move(temp_output_path, output_path)
                # Do NOT delete temp_dir here; wait until all processing is complete
                return_result = {
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
                break
            else:
                if similarity > best_similarity:
                    printq(f"    📈 New best attempt: {similarity:.1f}% (attempt {attempt + 1})")
                    best_similarity = similarity
                    # Keep track of best attempt data
                    best_verification = verification.copy()
                    best_verification["actual_duration"] = actual_audio_duration
                    # Save the best temp file path
                    best_audio_path = temp_output_path
                else:
                    printq(f"    📉 Lower quality than best attempt ({similarity:.1f}% < {best_similarity:.1f}%)")
                    # DEBUG: Do NOT delete temp_output_path
                    # if os.path.exists(temp_output_path):
                    #     os.remove(temp_output_path)
            
        except Exception as e:
            printq(f"    ❌ Generation failed: {e}")
            continue
    
    # Use best attempt if no attempt met threshold
    if best_audio_path and os.path.exists(best_audio_path):
        printq(f"  📋 No attempt met {confidence_threshold}% threshold. Using best attempt ({best_similarity:.1f}%)")
        
        # Move the best attempt to final location
        import shutil
        shutil.move(best_audio_path, output_path)
        # Do NOT delete temp_dir here; wait until all processing is complete
        return_result = {
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
    else:
        printq(f"  ❌ All {max_retries} attempts failed")
        return_result = None
    # Now, after all processing, delete temp_dir
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    return return_result


def main():
    parser = argparse.ArgumentParser(description='Enhanced audio generation with multilingual tokenizer')
    parser.add_argument('--input', required=True, help='Input JSON file with translations')
    parser.add_argument('--output', help='Output directory for audio files (default: translated_segments)')
    parser.add_argument('--language', required=True, help='Target language code (e.g., es, fr, de)')
    parser.add_argument('--speaker-dir', required=True, help='Directory containing speaker samples')
    parser.add_argument('--max-retries', type=int, default=5, help='Maximum retry attempts per segment')
    parser.add_argument('--confidence-threshold', type=float, default=85.0, help='Quality threshold percentage')
    parser.add_argument('--gpu-id', type=int, help='Specific GPU ID to use')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing (experimental)')
    parser.add_argument('--quiet', action='store_true', help='Suppress all output except warnings, errors, and key results')
    
    args = parser.parse_args()
    
    # Set default output directory if not specified
    if not args.output:
        args.output = 'translated_segments'
    
    # Initialize multilingual tokenizer
    printq(f"🧠 Initializing multilingual tokenizer for language: {args.language}")
    
    # Load translations
    translations = load_translations(args.input)
    if not translations:
        return False
    
    # Find speaker samples
    speaker_samples = find_speaker_samples(args.speaker_dir)
    if not speaker_samples:
        printq("Error: No speaker samples found")
        return False
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Setup GPU
    if args.parallel:
        device = setup_multi_gpu_processing()
        printq("Using parallel GPU processing")
    else:
        device = get_device(args.gpu_id)
        printq(f"Using single GPU: {device}")
    
    # Initialize XTTS model with maximum quality settings
    printq("🎵 Loading XTTS-v2 model with maximum quality settings...")
    try:
        # Set weights_only=False for PyTorch 2.6+ compatibility
        original_load = torch.load
        torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False) if 'weights_only' not in kwargs else original_load(*args, **kwargs)

        # Safe GPT2 patch for compatibility
        try:
            import transformers
            gpt2_model = getattr(transformers, "GPT2LMHeadModel", None)
            if gpt2_model and not hasattr(gpt2_model, "generate"):
                def generate_method(self, *args, **kwargs):
                    return super(type(self), self).generate(*args, **kwargs)
                gpt2_model.generate = generate_method
                printq("  🔧 Applied GPT2LMHeadModel.generate compatibility fix")
        except Exception as e:
            printq(f"  ⚠️  Could not apply GPT2 patch: {e}")
        
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        
        # Restore original torch.load
        torch.load = original_load
        
        if hasattr(tts, 'to') and device.type == 'cuda':
            tts = tts.to(device)
        
        printq("  ✅ TTS model loaded successfully with maximum quality settings:")
        printq("  • Temperature: 0.65 (stability)")
        printq("  • Repetition penalty: 5.0 (reduce repetition)")
        printq("  • Top-k: 20 (higher quality)")
        printq("  • Top-p: 0.75 (more predictable)")
        printq("  • Split sentences: enabled (better quality)")
        printq("  • Enhanced multilingual tokenizer: enabled")
        printq("  • Advanced anti-truncation: enabled")
        
    except Exception as e:
        printq(f"Error loading XTTS model: {e}")
        return False
    
    printq(f"🎯 Processing {len(translations['segments'])} segments...")
    
    # Process segments
    results = []
    successful = 0
    failed = 0
    skipped = 0
    
    printq.quiet = args.quiet

    # Restore: process all segments
    segments_to_process = translations['segments']

    for i, segment in enumerate(segments_to_process):
        # Only print separator and segment info if not quiet, or always if quiet
        if args.quiet:
            print("="*80)
            print(f"Processing segment {i+1}/{len(translations['segments'])}: {segment['segment_id']}")
        else:
            printq("\n" + "="*80)
            printq(f"Processing segment {i+1}/{len(translations['segments'])}: {segment['segment_id']}")
        
        # Skip very short segments
        if segment['duration'] < 0.5:
            printq(f"  ⏭️  Skipping short segment ({segment['duration']:.3f}s)")
            skipped += 1
            continue
        
        # Get speaker sample
        speaker = segment['speaker']
        if speaker not in speaker_samples:
            printq(f"  ❌ No speaker sample found for {speaker}")
            failed += 1
            continue
        
        speaker_sample = speaker_samples[speaker]
        
        # Generate audio
        result = generate_audio_segment(
            tts, segment, speaker_sample, args.language, args.output,
            args.max_retries, args.confidence_threshold,
            quiet=args.quiet
        )
        
        if result:
            result['segment_id'] = segment['segment_id']
            result['speaker'] = speaker
            result['expected_duration'] = segment['duration']
            results.append(result)
            successful += 1
            if args.quiet:
                print(f"    Quality: {result['whisper_confidence']:.1f}%")
                print(f"    Similarity: {result['similarity']:.1f}%")
            else:
                printq(f"    Quality: {result['whisper_confidence']:.1f}%")
                printq(f"    Similarity: {result['similarity']:.1f}%")
        else:
            failed += 1
            printq(f"  ❌ Failed after {args.max_retries} attempts")
    
    # Generate summary report
    printq(f"\n🎉 Audio generation completed!")
    printq(f"📊 Results Summary:")
    printq(f"  • Total segments processed: {len(translations['segments'])}")
    printq(f"  • Successful generations: {successful}")
    printq(f"  • Failed generations: {failed}")
    printq(f"  • Skipped segments (< 0.5s): {skipped}")
    
    if results:
        # Quality analysis
        similarities = [r['similarity'] for r in results]
        avg_similarity = sum(similarities) / len(similarities)
        passed_threshold = len([r for r in results if r['quality_status'] == 'passed_threshold'])
        
        printq(f"\n🎯 Quality Analysis:")
        printq(f"  • Average similarity: {avg_similarity:.1f}%")
        printq(f"  • Passed threshold (≥{args.confidence_threshold}%): {passed_threshold}")
        printq(f"  • Best below threshold: {len(results) - passed_threshold}")
        
        # Duration analysis
        expected_durations = [r['expected_duration'] for r in results if r.get('expected_duration')]
        actual_durations = [r['actual_duration'] for r in results if r.get('actual_duration')]
        
        if expected_durations and actual_durations:
            total_expected = sum(expected_durations)
            total_actual = sum(actual_durations)
            printq(f"\n📊 Duration Analysis:")
            printq(f"  • Total expected duration: {total_expected:.3f}s")
            printq(f"  • Total actual duration: {total_actual:.3f}s")
            printq(f"  • Duration ratio: {total_actual/total_expected:.3f}")
        
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
        
        printq(f"📁 Report saved to: {report_path}")
    
    # Clear GPU cache
    clear_gpu_cache(device)
    printq("🧹 GPU cache cleared")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 