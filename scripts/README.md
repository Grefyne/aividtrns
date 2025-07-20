# Audio Processing Pipeline Scripts

This directory contains all the Python scripts for the complete audio processing and translation workflow.

## Script Overview

### 1. `cleanup.py`
- **Purpose**: Removes previous processing files and directories to start fresh
- **Usage**: `python scripts/cleanup.py --force`
- **Output**: Cleans up all intermediate and output directories

### 2. `extract_audio.py`
- **Purpose**: Extracts audio from video files using ffmpeg
- **Usage**: `python scripts/extract_audio.py <video_path>`
- **Output**: WAV audio file in `audio_export/` directory

### 3. `speaker_audio_segmentation.py`
- **Purpose**: Performs speaker diarization using pyannote.audio
- **Usage**: `python scripts/speaker_audio_segmentation.py <audio_path>`
- **Output**: Speaker segments in `speaker_segments/` directory
- **Requirements**: HuggingFace token in `config/hf_token.txt`

### 4. `transcribe_segments.py`
- **Purpose**: Transcribes audio segments using Whisper
- **Usage**: `python scripts/transcribe_segments.py --segments_dir speaker_segments --model large`
- **Output**: Transcriptions in `transcribed_segments/` directory

### 5. `extract_speaker_samples.py`
- **Purpose**: Extracts individual speaker samples for voice cloning
- **Usage**: `python scripts/extract_speaker_samples.py <summary_path> <audio_path>`
- **Output**: Speaker samples in `speaker_samples/` directory

### 6. `translate_segments.py`
- **Purpose**: Translates transcribed segments using NLLB-200
- **Usage**: `python scripts/translate_segments.py <target_language>`
- **Output**: Translations in `translated_transcription/` directory

### 7. `generate_audio_with_tokenizer.py`
- **Purpose**: Generates translated audio using XTTS-v2 with Whisper verification
- **Usage**: `python scripts/generate_audio_with_tokenizer.py --language <lang> --input <translation_file>`
- **Output**: Generated audio in `translated_segments/` directory

### 8. `build_audio.py`
- **Purpose**: Merges translated audio segments into final audio file
- **Usage**: `python scripts/build_audio.py --transcription <file> --audio-dir <dir> --output <path>`
- **Output**: Merged audio file

### 9. `vocal_removal.py`
- **Purpose**: Removes vocals from original audio using Spleeter
- **Usage**: `python scripts/vocal_removal.py <audio_path>`
- **Output**: Background music in `vocal_removal_out/` directory

### 10. `final_audio.py`
- **Purpose**: Creates final audio mix with translated audio over background music
- **Usage**: `python scripts/final_audio.py`
- **Output**: Final mixed audio in `audio_export/final_audio.wav`

### 11. `face_area_detector.py`
- **Purpose**: Analyzes face areas in video for speaker positioning
- **Usage**: `python scripts/face_area_detector.py <video_path> --output-dir face_area`
- **Output**: Face area analysis in `face_area/` directory

### 12. `extract_face_areas.py`
- **Purpose**: Extracts face area videos and images for each speaker
- **Usage**: `python scripts/extract_face_areas.py <video_path> <face_area_results>`
- **Output**: Face area videos/images in `extracted_faces/` directory

### 13. `ls_train_data.py`
- **Purpose**: Prepares LatentSync v1.6 training data from extracted face areas
- **Usage**: `python scripts/ls_train_data.py --input_dir <dir> --output_dir latentsync_training_data`
- **Output**: Training data in `latentsync_training_data/` directory

## Complete Workflow

The scripts are designed to be run in sequence as part of the complete workflow:

1. **Cleanup** → Remove previous files
2. **Extract Audio** → Extract audio from video
3. **Speaker Segmentation** → Identify and segment speakers
4. **Transcribe** → Transcribe audio segments
5. **Extract Samples** → Get speaker voice samples
6. **Translate** → Translate transcriptions
7. **Generate Audio** → Create translated audio with TTS
8. **Build Audio** → Merge audio segments
9. **Vocal Removal** → Remove vocals from original
10. **Final Mix** → Combine translated audio with background
11. **Face Detection** → Analyze face areas in video
12. **Extract Faces** → Extract face area videos
13. **Training Data** → Prepare LatentSync training data

## Dependencies

All scripts require the dependencies listed in `../requirements.txt`:

- **Core**: torch, numpy, scipy
- **Audio**: pyannote.audio, whisper, TTS, spleeter
- **Translation**: transformers, sentencepiece
- **Vision**: opencv-python, matplotlib
- **Utilities**: tqdm, pathlib, json

## Configuration

- **HuggingFace Token**: Required for pyannote.audio, place in `config/hf_token.txt`
- **FFmpeg**: Required for audio/video processing
- **GPU**: Recommended for faster processing

## Error Handling

All scripts include comprehensive error handling and will:
- Validate input files and directories
- Provide detailed error messages
- Exit gracefully on failures
- Create detailed logs and summaries

## Output Structure

The pipeline creates a well-organized output structure:

```
├── audio_export/           # Extracted and final audio
├── speaker_segments/       # Speaker audio segments
├── transcribed_segments/   # Whisper transcriptions
├── speaker_samples/        # Individual speaker samples
├── translated_transcription/ # NLLB translations
├── translated_segments/    # Generated TTS audio
├── translated_audio/       # Merged translated audio
├── vocal_removal_out/      # Background music
├── face_area/             # Face area analysis
├── extracted_faces/        # Face area videos/images
└── latentsync_training_data/ # Training dataset
```

Each directory contains JSON summary files documenting the processing results. 