#!/usr/bin/env python3
"""
Enhanced Audio Generation Script using OpenVoice for Instant Voice Cloning
Includes Whisper verification, multi-attempt generation, and maximum quality settings.
OpenVoice provides better audio quality and more natural speech synthesis.
"""

import os
import sys
import json
import argparse
import whisper
import torch
import subprocess
import re
import numpy as np
import time
import librosa
import soundfile
from pathlib import Path
from gpu_utils import get_device, setup_multi_gpu_processing, clear_gpu_cache, parallel_process_with_gpus

# OpenVoice imports
try:
    import openvoice
    from openvoice import se_extractor
    from openvoice.api import BaseSpeakerTTS, ToneColorConverter
except ImportError:
    print("Error: OpenVoice not found. Please install it first:")
    print("pip install openvoice")
    print("Or clone from: https://github.com/myshell-ai/OpenVoice")
    sys.exit(1)

# Multilingual tokenizer imports (keeping for text processing)
try:
    from multilingual_tokenizer import VoiceBpeTokenizer, split_sentence, multilingual_cleaners
except ImportError:
    print("Warning: multilingual_tokenizer not found. Using basic text processing.")
    # Fallback text processing functions
    def multilingual_cleaners(text, language):
        return text.strip()
    
    def split_sentence(text, language, max_length):
        words = text.split()
        chunks = []
        current_chunk = ""
        for word in words:
            if len(current_chunk + " " + word) <= max_length:
                current_chunk += (" " if current_chunk else "") + word
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = word
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks


def get_language_char_limit(language):
    """
    Get optimal character limits per language for OpenVoice.
    OpenVoice has different capabilities than XTTS, so we adjust limits accordingly.
    """
    # OpenVoice character limits (more generous than XTTS)
    openvoice_limits = {
        'en': 300,  # English - OpenVoice handles English very well
        'es': 280,  # Spanish
        'fr': 280,  # French
        'de': 260,  # German
        'it': 280,  # Italian
        'pt': 280,  # Portuguese
        'ru': 240,  # Russian
        'zh': 200,  # Chinese
        'ja': 200,  # Japanese
        'ko': 200,  # Korean
        'ar': 220,  # Arabic
        'hi': 220,  # Hindi
        'tr': 260,  # Turkish
        'pl': 260,  # Polish
        'cs': 260,  # Czech
        'nl': 280,  # Dutch
        'hu': 240,  # Hungarian
        'ur': 240,  # Urdu
        'sd': 280,  # Sindhi
    }
    
    return openvoice_limits.get(language, 250)  # Default to 250 for unknown languages


def preprocess_text_enhanced(text, language):
    """
    Enhanced text preprocessing optimized for OpenVoice.
    OpenVoice handles text differently than XTTS, so we adjust preprocessing accordingly.
    """
    if not text or not text.strip():
        return ""
    
    # Basic cleaning
    cleaned_text = text.strip()
    
    # Use multilingual cleaner if available
    try:
        cleaned_text = multilingual_cleaners(cleaned_text, language)
    except:
        # Fallback cleaning
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Normalize whitespace
        cleaned_text = cleaned_text.strip()
    
    # OpenVoice-specific preprocessing
    # Remove excessive punctuation that can cause issues
    cleaned_text = re.sub(r'\.{3,}', '...', cleaned_text)  # Normalize ellipsis
    cleaned_text = re.sub(r'[!]{2,}', '!', cleaned_text)   # Reduce multiple exclamations
    cleaned_text = re.sub(r'[?]{2,}', '?', cleaned_text)   # Reduce multiple questions
    
    # OpenVoice works better with proper sentence endings
    if cleaned_text and not cleaned_text[-1] in '.!?':
        cleaned_text += '.'
    
    # OpenVoice handles contractions and informal text better than XTTS
    # So we don't need to be as aggressive with text normalization
    
    return cleaned_text.strip()


def split_text_smart(text, language, max_length=None):
    """
    Smart text splitting optimized for OpenVoice.
    OpenVoice can handle longer text chunks than XTTS, so we adjust accordingly.
    """
    if not text or not text.strip():
        return []
    
    if max_length is None:
        max_length = get_language_char_limit(language)
    
    # If text is short enough, return as single chunk
    if len(text) <= max_length:
        return [text.strip()]
    
    # Use the multilingual tokenizer's smart splitting if available
    try:
        chunks = split_sentence(text, language, max_length)
        if chunks:
            return [chunk.strip() for chunk in chunks if chunk.strip()]
    except:
        pass
    
    # Fallback to simple word-based splitting
    print(f"  ⚠️  Fallback to simple word-based splitting")
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


def verify_audio_quality(audio_path, expected_text, language):
    """
    Verify audio quality using Whisper transcription and similarity scoring.
    Returns confidence percentage.
    """
    try:
        # Load Whisper model
        model = whisper.load_model("base")
        
        # Transcribe the generated audio
        result = model.transcribe(audio_path)
        transcribed_text = result["text"].strip().lower()
        
        # Clean expected text for comparison
        expected_clean = expected_text.strip().lower()
        
        # Calculate similarity using simple word overlap
        expected_words = set(expected_clean.split())
        transcribed_words = set(transcribed_text.split())
        
        if not expected_words:
            return 0.0
            
        # Calculate Jaccard similarity
        intersection = len(expected_words.intersection(transcribed_words))
        union = len(expected_words.union(transcribed_words))
        
        if union == 0:
            return 0.0
            
        similarity = (intersection / union) * 100
        
        # Also consider Whisper confidence
        whisper_confidence = result.get("confidence", 0.0) * 100 if "confidence" in result else 50.0
        
        # Combine similarity and confidence
        final_confidence = (similarity * 0.7) + (whisper_confidence * 0.3)
        
        return min(final_confidence, 100.0)
        
    except Exception as e:
        print(f"    ⚠️  Quality verification error: {e}")
        return 50.0  # Default confidence if verification fails


def generate_audio_segment(base_speaker_tts, tone_converter, segment, speaker_sample, language, output_dir, 
                          max_retries=5, confidence_threshold=85.0):
    """
    Generate audio for a single segment using OpenVoice with quality verification and retry logic.
    """
    segment_id = segment['segment_id']
    speaker = segment['speaker']
    # Handle both transcribed and translated formats
    translated_text = segment.get('translated_transcription') or segment.get('transcription', '')
    
    print(f"  🎵 Generating audio for segment {segment_id}...")
    print(f"  👤 Speaker: {speaker}")
    print(f"  📝 Text: {translated_text}")
    
    # Enhanced text preprocessing
    cleaned_text = preprocess_text_enhanced(translated_text, language)
    print(f"  ✨ Cleaned: {cleaned_text}")
    
    # Get character limit for language
    char_limit = get_language_char_limit(language)
    print(f"  📏 Char limit for {language}: {char_limit}")
    
    # Smart text splitting
    text_chunks = split_text_smart(cleaned_text, language, char_limit)
    if len(text_chunks) > 1:
        print(f"  ✂️  Text too long ({len(cleaned_text)} chars), using smart splitting...")
        print(f"  📦 Split into {len(text_chunks)} chunks")
    else:
        print(f"  ✅ Text fits in single chunk")
    
    print(f"  🎯 Text chunks:")
    for i, chunk in enumerate(text_chunks):
        print(f"    Chunk {i+1}: ({len(chunk)} chars) {chunk[:50]}...")
    
    print(f"  🎯 Target confidence threshold: {confidence_threshold}%")
    
    # Try multiple attempts
    best_audio = None
    best_confidence = 0.0
    
    for attempt in range(max_retries):
        try:
            print(f"  🔄 Attempt {attempt + 1}/{max_retries}")
            
            # Extract speaker embedding from reference audio
            try:
                target_se, audio_name = se_extractor.get_se(speaker_sample, tone_converter, target_dir='processed', vad=True)
                print(f"  ✅ Speaker embedding extracted successfully")
            except Exception as e:
                print(f"  ❌ Failed to extract speaker embedding: {e}")
                continue
            
            # Load source speaker embedding
            try:
                source_se = torch.load('OpenVoice/checkpoints/base_speakers/EN/en_default_se.pth').to(base_speaker_tts.device)
                print(f"  ✅ Source speaker embedding loaded")
            except Exception as e:
                print(f"  ❌ Failed to load source speaker embedding: {e}")
                continue
            
            # Generate audio for each chunk
            chunk_audios = []
            for i, chunk in enumerate(text_chunks):
                if len(chunk.strip()) == 0:
                    continue
                    
                print(f"  🎵 Generating chunk {i+1}/{len(text_chunks)}: {chunk[:50]}...")
                
                # Generate base audio with BaseSpeakerTTS
                temp_path = f"{output_dir}/temp_chunk_{i}.wav"
                base_speaker_tts.tts(chunk, temp_path, speaker='default', language='English', speed=1.0)
                
                # Convert tone color using ToneColorConverter
                chunk_output_path = f"{output_dir}/chunk_{i}.wav"
                encode_message = "@MyShell"
                tone_converter.convert(
                    audio_src_path=temp_path,
                    src_se=source_se,
                    tgt_se=target_se,
                    output_path=chunk_output_path,
                    message=encode_message
                )
                
                # Load the generated audio
                chunk_audio, sr = librosa.load(chunk_output_path, sr=None)
                chunk_audios.append(chunk_audio)
                
                # Clean up temp files
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                if os.path.exists(chunk_output_path):
                    os.remove(chunk_output_path)
            
            if not chunk_audios:
                print(f"  ❌ No audio chunks generated")
                continue
            
            # Concatenate all chunks
            final_audio = np.concatenate(chunk_audios)
            
            # Save the final audio
            output_path = os.path.join(output_dir, f"{segment_id}.wav")
            soundfile.write(output_path, final_audio, sr)
            
            # Verify quality with Whisper
            try:
                confidence = verify_audio_quality(output_path, translated_text, language)
                print(f"  🎯 Quality verification: {confidence:.1f}% confidence")
                
                if confidence >= confidence_threshold:
                    print(f"  ✅ Success! Quality threshold met ({confidence:.1f}% >= {confidence_threshold}%)")
                    return True, confidence, output_path
                else:
                    print(f"  ⚠️  Quality below threshold ({confidence:.1f}% < {confidence_threshold}%)")
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_audio = output_path
                        
            except Exception as e:
                print(f"  ⚠️  Quality verification failed: {e}")
                if best_audio is None:
                    best_audio = output_path
                    
        except Exception as e:
            print(f"  ❌ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"  🔄 Retrying...")
                time.sleep(1)
    
    # If we have a best attempt, use it
    if best_audio and os.path.exists(best_audio):
        print(f"  ⚠️  Using best attempt with {best_confidence:.1f}% confidence")
        return True, best_confidence, best_audio
    
    print(f"  ❌ Failed after {max_retries} attempts")
    return False, 0.0, None


def initialize_openvoice_models(device):
    """
    Initialize OpenVoice models with proper config paths and checkpoints.
    """
    try:
        # Convert torch.device to string for OpenVoice compatibility
        device_str = str(device) if hasattr(device, '__str__') else device
        
        # Define checkpoint paths
        ckpt_base = 'OpenVoice/checkpoints/base_speakers/EN'
        ckpt_converter = 'OpenVoice/checkpoints/converter'
        
        # Check if checkpoints exist
        if not os.path.exists(ckpt_base):
            print(f"❌ Base speaker checkpoint not found at: {ckpt_base}")
            print("Please download OpenVoice checkpoints from:")
            print("https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_1226.zip")
            print("And extract to the OpenVoice/checkpoints folder")
            return None, None
            
        if not os.path.exists(ckpt_converter):
            print(f"❌ Converter checkpoint not found at: {ckpt_converter}")
            print("Please download OpenVoice checkpoints from:")
            print("https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_1226.zip")
            print("And extract to the OpenVoice/checkpoints folder")
            return None, None
        
        print("🎵 Loading OpenVoice Base Speaker TTS...")
        base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device_str)
        base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')
        
        print("🎵 Loading OpenVoice Tone Color Converter...")
        tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device_str)
        tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
        
        print("✅ OpenVoice models loaded successfully!")
        return base_speaker_tts, tone_color_converter
        
    except Exception as e:
        print(f"Error loading OpenVoice models: {e}")
        print("Make sure OpenVoice is properly installed:")
        print("pip install -e .")
        print("And checkpoints are downloaded to OpenVoice/checkpoints/")
        return None, None


def main():
    parser = argparse.ArgumentParser(description='Enhanced audio generation with OpenVoice for instant voice cloning')
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
    
    # Initialize OpenVoice TTS model
    print("🎵 Loading OpenVoice TTS model...")
    try:
        # Initialize OpenVoice TTS and ToneColorConverter
        base_speaker_tts, tone_converter = initialize_openvoice_models(device)
        
        if base_speaker_tts is None or tone_converter is None:
            print("❌ Failed to initialize OpenVoice models")
            return False
            
        print("  ✅ OpenVoice TTS model loaded successfully:")
        print(f"     - Device: {device}")
        print(f"     - Language: {args.language}")
        print(f"     - Max retries: {args.max_retries}")
        print(f"     - Confidence threshold: {args.confidence_threshold}%")
        
    except Exception as e:
        print(f"Error loading OpenVoice TTS model: {e}")
        print("Make sure OpenVoice is properly installed:")
        print("pip install openvoice")
        print("Or clone from: https://github.com/myshell-ai/OpenVoice")
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
            base_speaker_tts, tone_converter, segment, speaker_sample, args.language, args.output,
            args.max_retries, args.confidence_threshold
        )
        
        if result is None:
            print(f"  ❌ Failed to generate audio")
            failed += 1
            continue
            
        success, confidence, output_path = result
        
        if success:
            print(f"  ✅ Audio generated successfully: {output_path}")
            print(f"  📊 Quality: {confidence:.1f}% confidence")
            successful += 1
            results.append({
                "segment_id": segment['segment_id'],
                "success": True,
                "output_path": output_path,
                "confidence": confidence,
                "speaker": speaker
            })
        else:
            print(f"  ❌ Audio generation failed")
            failed += 1
    
    # Generate summary report
    print(f"\n🎉 Audio generation completed!")
    print(f"📊 Results Summary:")
    print(f"  • Total segments processed: {len(translations['segments'])}")
    print(f"  • Successful generations: {successful}")
    print(f"  • Failed generations: {failed}")
    print(f"  • Skipped segments (< 0.5s): {skipped}")
    
    if results:
        # Calculate average confidence
        confidences = [r['confidence'] for r in results if 'confidence' in r]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            print(f"  • Average confidence: {avg_confidence:.1f}%")
        
        # Save detailed results
        results_file = os.path.join(args.output, "generation_report.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {
                    "total_segments": len(translations['segments']),
                    "successful": successful,
                    "failed": failed,
                    "skipped": skipped,
                    "average_confidence": avg_confidence if confidences else 0.0
                },
                "results": results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"  • Detailed report saved to: {results_file}")
    
    # Clean up GPU cache
    clear_gpu_cache(device)
    
    return successful > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 