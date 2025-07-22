import os
import json
import logging
import multiprocessing
import tempfile
import shutil
import difflib
import string
from pydub import AudioSegment
from pydub.silence import detect_silence
from TTS.api import TTS
import whisper
import torch
import textwrap
from tokenizer import multilingual_cleaners, split_sentence  # Import fixed tokenizer

# Optional: Disable cuDNN to test if warnings affect pauses
# torch.backends.cudnn.enabled = False

# Supported XTTS v2 languages
xtts_languages = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"]

# Language mappings from three-letter to two-letter codes
three_to_two = {
    "eng": "en", "spa": "es", "fra": "fr", "deu": "de", "por": "pt", "ita": "it",
    "pol": "pl", "ara": "ar", "zho": "zh-cn", "ces": "cs", "rus": "ru", "nld": "nl",
    "tur": "tr", "hun": "hu", "kor": "ko", "jpn": "ja", "hin": "hi", "urd": "ur", "snd": "sd"
}

# Character limits for splitting text by language
char_limits = {
    "en": 250, "es": 239, "fr": 273, "de": 253, "it": 213, "pt": 203, "pl": 224,
    "zh-cn": 82, "ar": 166, "cs": 186, "ru": 182, "nl": 251, "tr": 226, "ja": 71,
    "hu": 224, "ko": 95, "hi": 150, "ur": 150, "sd": 400
}

def remove_punctuation(text):
    """Remove punctuation from text for comparison."""
    return text.translate(str.maketrans('', '', string.punctuation))

def fallback_split(text, char_limit):
    """Fallback splitting mechanism if split_sentence fails."""
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            if len(current_chunk) + len(sentence) + 1 <= char_limit:
                current_chunk += (". " + sentence) if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if len(sentence) > char_limit:
                    for line in textwrap.wrap(
                        sentence,
                        width=char_limit,
                        drop_whitespace=True,
                        break_on_hyphens=False,
                        tabsize=1
                    ):
                        chunks.append(line.strip())
                else:
                    chunks.append(sentence)
                current_chunk = ""
    if current_chunk:
        chunks.append(current_chunk)
    return chunks if chunks else [text]

def trim_silence(audio_segment, min_silence_len=30, silence_thresh=-60):
    """Trim silence from the start and end of an audio segment."""
    silences = detect_silence(audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    if not silences:
        return audio_segment
    start = silences[0][0] if silences and silences[0][0] == 0 else 0
    end = silences[-1][1] if silences and silences[-1][1] == len(audio_segment) else len(audio_segment)
    return audio_segment[start:end]

def adjust_audio_duration(audio, target_duration, current_duration):
    """Adjust audio duration to match target duration by speeding up or slowing down."""
    if target_duration and abs(current_duration - target_duration) / target_duration > 0.05:  # 5% tolerance
        speed_factor = current_duration / target_duration
        if 0.9 <= speed_factor <= 1.1:  # Limit to avoid distortion
            audio = audio.speedup(playback_speed=speed_factor, crossfade=30)
    return audio

def worker(task_queue, result_queue, gpu_id, speaker_samples_dir, output_dir, source):
    """Worker function to process segments on a specific GPU."""
    try:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        tts = TTS(model_name="xtts_v2").to(device)
        whisper_model = whisper.load_model("base", device=device)

        logger = logging.getLogger(f"Process-{gpu_id}")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(f"Process-{gpu_id}: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        while True:
            segment = task_queue.get()
            if segment is None:
                break
            try:
                segment_id = segment['segment_id']
                text_key = 'translated_transcription' if source == 'translated' else 'transcription'
                text = segment.get(text_key)
                language = segment.get('language', 'en') if source == 'transcribed' else segment.get('target_language_code', 'spa_Latn')
                if not text:
                    raise ValueError(f"No {text_key} provided")
                if not language:
                    raise ValueError("Language not specified")
                
                lang = three_to_two.get(language.lower().split('_')[0], language.lower().split('_')[0])
                if lang not in xtts_languages:
                    logger.warning(f"Language {lang} not supported by XTTS v2, defaulting to 'en'")
                    lang = "en"
                language_for_tts = lang
                
                speaker = segment['speaker']
                expected_duration = segment['duration']
                audio_file = segment['audio_file']
                speaker_id = speaker.split('_')[1]
                speaker_sample = os.path.join(speaker_samples_dir, f"speaker{speaker_id}_sample.wav")

                logger.info(f"Processing segment {segment_id}, Speaker: {speaker}, Language: {lang}, Text: {text}")

                try:
                    text_clean = multilingual_cleaners(text, lang)
                except Exception as e:
                    logger.error(f"Failed to clean text for segment {segment_id}: {e}")
                    text_clean = text  # Fallback to raw text
                char_limit = char_limits.get(lang, 250)
                if len(text_clean) < 50:  # Skip splitting for short texts
                    chunks = [text_clean]
                else:
                    try:
                        chunks = split_sentence(text_clean, lang, char_limit)
                    except Exception as e:
                        logger.warning(f"split_sentence failed for segment {segment_id}: {e}. Using fallback splitting.")
                        chunks = fallback_split(text_clean, char_limit)
                logger.info(f"Segment {segment_id} split into {len(chunks)} chunks: {chunks}")

                temp_dir = tempfile.mkdtemp(prefix=f"segment_{segment_id}_")
                best_similarity = 0
                best_audio = None
                best_transcribed = None
                best_whisper_confidence = 0
                silence_info = []

                for attempt in range(1, 6):
                    chunk_files = []
                    audio = AudioSegment.empty()
                    try:
                        for i, chunk in enumerate(chunks):
                            chunk_file = os.path.join(temp_dir, f"chunk_{segment_id}_{i}.wav")
                            # Dynamic speed based on expected duration
                            speed = 1.0
                            if expected_duration:
                                chunk_duration = expected_duration / max(1, len(chunks))
                                words_per_second = len(chunk.split()) / chunk_duration
                                speed = max(0.9, min(1.1, words_per_second / 5.0))  # Tighter bounds
                            tts.tts_to_file(
                                text=chunk,
                                speaker_wav=speaker_sample,
                                language=language_for_tts,
                                file_path=chunk_file,
                                split_sentences=False,
                                speed=speed
                            )
                            chunk_audio = AudioSegment.from_wav(chunk_file)
                            chunk_silences = detect_silence(chunk_audio, min_silence_len=30, silence_thresh=-60)
                            silence_info.append({f"chunk_{i}": [(start/1000.0, end/1000.0, (end-start)/1000.0) for start, end in chunk_silences]})
                            chunk_audio = trim_silence(chunk_audio, min_silence_len=30, silence_thresh=-60)
                            chunk_files.append(chunk_file)
                            if i > 0:
                                audio = audio.append(chunk_audio, crossfade=min(30, len(chunk_audio)//2))
                            else:
                                audio = chunk_audio
                        
                        # Trim silence from final audio
                        final_silences = detect_silence(audio, min_silence_len=30, silence_thresh=-60)
                        silence_info.append({"final_audio": [(start/1000.0, end/1000.0, (end-start)/1000.0) for start, end in final_silences]})
                        audio = trim_silence(audio, min_silence_len=30, silence_thresh=-60)
                        # Adjust duration
                        current_duration = len(audio) / 1000.0
                        if expected_duration:
                            audio = adjust_audio_duration(audio, expected_duration, current_duration)

                        temp_output = os.path.join(temp_dir, f"temp_{segment_id}.wav")
                        audio.export(temp_output, format="wav")

                        if len(text_clean.split()) < 3:
                            best_similarity = 1.0
                            best_audio = audio
                            best_transcribed = text_clean
                            best_whisper_confidence = 0
                            logger.info(f"Segment {segment_id}, Attempt {attempt}: Skipped verification (text too short)")
                            break

                        result = whisper_model.transcribe(temp_output, language=lang)
                        transcribed_text = result["text"]
                        expected_clean = remove_punctuation(text_clean).lower()
                        transcribed_clean = remove_punctuation(transcribed_text).lower()
                        similarity = difflib.SequenceMatcher(None, expected_clean, transcribed_clean).ratio()

                        if "segments" in result and result["segments"]:
                            confidences = [seg.get("confidence", 0) for seg in result["segments"]]
                            whisper_confidence = sum(confidences) / len(confidences) if confidences else 0
                        else:
                            whisper_confidence = 0

                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_audio = audio
                            best_transcribed = transcribed_text
                            best_whisper_confidence = whisper_confidence
                        
                        logger.info(f"Segment {segment_id}, Attempt {attempt}: Similarity {similarity*100:.1f}%, Transcribed: {transcribed_text[:100]}...")
                        logger.info(f"Segment {segment_id}, Attempt {attempt}: Silence Info: {silence_info[-2:]}")
                        if best_similarity >= 0.85 or (len(text_clean.split()) < 5 and best_similarity >= 0.5):
                            break
                    except Exception as e:
                        logger.error(f"Error in attempt {attempt} for segment {segment_id}: {e}")
                    finally:
                        for chunk_file in chunk_files:
                            if os.path.exists(chunk_file):
                                os.remove(chunk_file)

                if best_audio is None:
                    raise ValueError("No audio generated for segment")
                final_output_path = os.path.join(output_dir, audio_file)
                os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
                best_audio.export(final_output_path, format="wav")
                actual_duration = len(best_audio) / 1000.0
                quality_status = "good" if best_similarity >= 0.85 else "best_below_threshold"

                report_info = {
                    "success": True,
                    "output_path": final_output_path,
                    "similarity": best_similarity * 100,
                    "whisper_confidence": best_whisper_confidence,
                    "attempts": attempt,
                    "text_chunks": len(chunks),
                    "combined_text": text_clean,
                    "transcribed_text": best_transcribed,
                    "actual_duration": actual_duration,
                    "quality_status": quality_status,
                    "segment_id": segment_id,
                    "speaker": speaker,
                    "expected_duration": expected_duration,
                    "language": lang,
                    "silence_info": silence_info
                }
                logger.info(f"Segment {segment_id} completed: Similarity {best_similarity*100:.1f}%, Duration {actual_duration}s, Output: {final_output_path}")
                result_queue.put(report_info)
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.error(f"Error processing segment {segment_id}: {e}")
                result_queue.put({"success": False, "segment_id": segment_id, "error": str(e)})
    except Exception as e:
        logger.error(f"Worker on GPU {gpu_id} failed to initialize: {e}")
        result_queue.put({"success": False, "segment_id": "worker_init", "error": str(e)})

def main(input_json, speaker_samples_dir, output_dir, source):
    """Main function to process the transcript and generate audio files."""
    if source not in ["transcribed", "translated"]:
        raise ValueError("Source must be 'transcribed' or 'translated'")

    logging.basicConfig(level=logging.INFO, format="Main: %(message)s")
    logger = logging.getLogger("Main")

    try:
        with open(input_json, 'r') as f:
            data = json.load(f)
        segments = data.get('segments', data) if isinstance(data, dict) else data
    except Exception as e:
        logger.error(f"Failed to load transcript {input_json}: {e}")
        return

    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    for segment in segments:
        task_queue.put(segment)
    num_workers = min(2, torch.cuda.device_count())
    for _ in range(num_workers):
        task_queue.put(None)

    workers = []
    for gpu_id in range(num_workers):
        p = multiprocessing.Process(
            target=worker,
            args=(task_queue, result_queue, gpu_id, speaker_samples_dir, output_dir, source)
        )
        p.start()
        workers.append(p)

    results = []
    for _ in range(len(segments)):
        try:
            result = result_queue.get(timeout=120)
            if not result["success"]:
                logger.error(f"Error in segment {result['segment_id']}: {result['error']}")
            results.append(result)
        except multiprocessing.queues.Empty:
            logger.error("Timeout waiting for result; a worker may have failed.")
            break

    for p in workers:
        p.join()

    report = {"segments": results}
    report_path = os.path.join(output_dir, "generation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    logger.info(f"Report generated at {report_path}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    torch.set_num_threads(1)

    transcript_json = "transcribed_segments/transcribed_segments.json"
    translated_json = "translated_transcription/translated_transcription.json"
    speaker_samples_dir = "speaker_samples"
    output_dir = "translated_audio"

    source = "transcribed"  # Change to "translated" for translated transcriptions
    input_json = transcript_json if source == "transcribed" else translated_json

    os.makedirs(output_dir, exist_ok=True)
    main(input_json, speaker_samples_dir, output_dir, source)