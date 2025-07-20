#!/usr/bin/env python3
"""
Segment translation script using NLLB (No Language Left Behind).
Translates transcribed segments to target language.
"""

import os
import sys
import json
import argparse
import torch
import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from gpu_utils import get_device, setup_multi_gpu_processing, clear_gpu_cache, parallel_process_with_gpus


def load_transcriptions(input_file):
    """Load transcribed segments from JSON file."""
    if not os.path.exists(input_file):
        print(f"Error: Transcription file not found: {input_file}")
        return None
    
    with open(input_file, 'r') as f:
        return json.load(f)


def split_into_sentences(text):
    """Split text into sentences using regex."""
    # Split on sentence endings followed by space or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out empty sentences and clean up
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def get_nllb_language_code(language):
    """Get NLLB language code from language name."""
    language_map = {
        "spanish": "spa_Latn",
        "french": "fra_Latn", 
        "german": "deu_Latn",
        "italian": "ita_Latn",
        "portuguese": "por_Latn",
        "russian": "rus_Cyrl",
        "chinese": "zho_Hans",
        "japanese": "jpn_Jpan",
        "korean": "kor_Hang",
        "arabic": "arb_Arab",
        "hindi": "hin_Deva",
        "bengali": "ben_Beng",
        "dutch": "nld_Latn",
        "swedish": "swe_Latn",
        "norwegian": "nob_Latn",
        "danish": "dan_Latn",
        "finnish": "fin_Latn",
        "polish": "pol_Latn",
        "czech": "ces_Latn",
        "hungarian": "hun_Latn",
        "romanian": "ron_Latn",
        "bulgarian": "bul_Cyrl",
        "greek": "ell_Grek",
        "turkish": "tur_Latn",
        "hebrew": "heb_Hebr",
        "thai": "tha_Thai",
        "vietnamese": "vie_Latn",
        "indonesian": "ind_Latn",
        "malay": "msa_Latn",
        "filipino": "fil_Latn",
        "urdu": "urd_Arab",
        "persian": "pes_Arab",
        "swahili": "swh_Latn",
        "yoruba": "yor_Latn",
        "igbo": "ibo_Latn",
        "hausa": "hau_Latn",
        "amharic": "amh_Ethi",
        "somali": "som_Latn",
        "zulu": "zul_Latn",
        "xhosa": "xho_Latn",
        "afrikaans": "afr_Latn"
    }
    
    language_lower = language.lower()
    if language_lower in language_map:
        return language_map[language_lower]
    else:
        print(f"Warning: Unknown language '{language}'. Using default code.")
        return f"{language_lower}_Latn"


def translate_segments(transcriptions, target_language, output_dir, gpu_id=None, use_parallel=False):
    """Translate transcribed segments using NLLB."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get NLLB language code
    target_lang_code = get_nllb_language_code(target_language)
    print(f"Target language: {target_language} (NLLB code: {target_lang_code})")
    
    # Setup GPU processing
    available_gpus = setup_multi_gpu_processing()
    if available_gpus:
        if gpu_id is None:
            gpu_id = available_gpus[0]  # Use first available GPU
        device = get_device(gpu_id)
        print(f"Using GPU {gpu_id} for NLLB processing")
    else:
        device = torch.device("cpu")
        print("Using CPU for NLLB processing")
    
    # Load NLLB model and tokenizer
    print("Loading NLLB model...")
    try:
        model_name = "facebook/nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        # Move model to device
        model = model.to(device)
    except Exception as e:
        print(f"Error loading NLLB model: {e}")
        return False
    
    print(f"Translating {len(transcriptions['segments'])} segments...")
    
    # For now, use sequential processing to avoid pickling issues with models
    # TODO: Implement proper parallel processing with model loading per process
    translated_segments = []
    
    for i, segment in enumerate(transcriptions['segments']):
        result = _translate_single_segment(segment, tokenizer, model, device, target_lang_code, target_language, model_name, i+1, len(transcriptions['segments']))
        if result:
            translated_segments.append(result)
    
    # Save translations
    output_data = {
        "model_used": model_name,
        "source_language": "en",
        "target_language": target_language,
        "target_language_code": target_lang_code,
        "total_segments": len(translated_segments),
        "segments": translated_segments
    }
    
    output_path = os.path.join(output_dir, "translated_transcription.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nTranslation completed!")
    print(f"Translated segments: {len(translated_segments)}/{len(transcriptions['segments'])}")
    print(f"Output saved to: {output_path}")
    
    # Clear GPU cache
    if available_gpus:
        clear_gpu_cache(gpu_id)
    
    return True


def _translate_single_segment(segment, tokenizer, model, device, target_lang_code, target_language, model_name, segment_num=None, total_segments=None):
    """Translate a single segment by chunking into sentences."""
    if segment_num and total_segments:
        print(f"Translating segment {segment_num}/{total_segments}: {segment['segment_id']}")
    else:
        print(f"Translating segment: {segment['segment_id']}")
    
    original_text = segment['transcription']
    if not original_text.strip():
        print(f"  Skipping empty transcription for {segment['segment_id']}")
        return None
    
    try:
        # Split text into sentences
        sentences = split_into_sentences(original_text)
        if not sentences:
            print(f"  No valid sentences found for {segment['segment_id']}")
            return None
        
        print(f"  Processing {len(sentences)} sentences")
        
        # Translate each sentence
        translated_sentences = []
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            try:
                # Tokenize input text for NLLB
                max_input_length = 512
                max_output_length = 512
                
                inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=max_input_length)
                
                # Move inputs to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generate translation with NLLB using transformers pipeline method
                # Set the source and target language tokens
                tokenizer.src_lang = "eng_Latn"
                tokenizer.tgt_lang = target_lang_code
                
                translated_tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang_code),
                    max_length=max_output_length,
                    num_beams=5,
                    early_stopping=True,
                    do_sample=False
                )
                
                # Decode translation
                translated_sentence = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                translated_sentences.append(translated_sentence)
                
                if segment_num and total_segments:
                    print(f"    Sentence {i+1}: {sentence}")
                    print(f"    Translated: {translated_sentence}")
                    
            except Exception as e:
                print(f"    Error translating sentence {i+1}: {e}")
                # Keep original sentence if translation fails
                translated_sentences.append(sentence)
        
        # Join translated sentences
        translated_text = " ".join(translated_sentences)
        
        # Create translated segment entry
        translated_entry = {
            "segment_id": segment['segment_id'],
            "speaker": segment['speaker'],
            "start_time": segment['start_time'],
            "end_time": segment['end_time'],
            "duration": segment['duration'],
            "start_time_str": segment['start_time_str'],
            "end_time_str": segment['end_time_str'],
            "duration_str": segment['duration_str'],
            "audio_file": segment['audio_file'],
            "original_transcription": original_text,
            "translated_transcription": translated_text,
            "source_language": segment.get('language', 'en'),
            "target_language": target_language,
            "target_language_code": target_lang_code
        }
        
        if segment_num and total_segments:
            print(f"  Original: {original_text}")
            print(f"  Translated: {translated_text}")
        
        return translated_entry
        
    except Exception as e:
        print(f"  Error translating {segment['segment_id']}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Translate transcribed segments using NLLB")
    parser.add_argument("target_language", help="Target language for translation")
    parser.add_argument("--input_file", default="transcribed_segments/transcribed_segments.json",
                       help="Path to transcribed segments JSON file")
    parser.add_argument("--output_dir", default="translated_transcription",
                       help="Output directory for translations")
    parser.add_argument("--gpu-id", type=int, default=None,
                       help="Specific GPU ID to use (default: auto-select)")
    parser.add_argument("--parallel", action="store_true",
                       help="Use parallel processing across multiple GPUs")
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Load transcriptions
    transcriptions = load_transcriptions(args.input_file)
    if not transcriptions:
        sys.exit(1)
    
    # Perform translation
    success = translate_segments(transcriptions, args.target_language, args.output_dir, args.gpu_id, args.parallel)
    
    if not success:
        print("Translation failed!")
        sys.exit(1)
    else:
        print("Translation completed successfully!")


if __name__ == "__main__":
    main() 