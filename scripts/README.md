# Audio Processing Scripts

This directory contains GPU-accelerated scripts for comprehensive audio processing including speaker segmentation, transcription, translation, and audio generation with multilingual support.

## Core Processing Scripts

### 1. **speaker_audio_segmentation.py**
Performs speaker diarization using pyannote.audio.

**Features:**
- GPU-accelerated speaker diarization
- Automatic best GPU selection
- Exports segmented audio files per speaker
- Creates detailed segmentation summaries

**Usage:**
```bash
python speaker_audio_segmentation.py <audio_file> <output_dir> [--gpu-id GPU_ID]
```

### 2. **transcribe_segments.py**
Transcribes audio segments using Whisper.

**Features:**
- GPU-accelerated Whisper transcription
- Parallel processing across multiple GPUs
- Automatic best GPU selection
- Supports various Whisper model sizes

**Usage:**
```bash
python transcribe_segments.py [--model MODEL] [--gpu-id GPU_ID] [--parallel]
```

### 3. **translate_segments.py**
Translates transcriptions using NLLB model.

**Features:**
- GPU-accelerated NLLB translation
- Sentence-level chunking for better quality
- Automatic best GPU selection
- Proper UTF-8 character encoding
- Support for multiple target languages

**Usage:**
```bash
python translate_segments.py <target_language> [--gpu-id GPU_ID] [--parallel]
```

### 4. **generate_audio_with_tokenizer.py**
Generates translated audio using XTTS-v2 with advanced text preprocessing.

**Features:**
- GPU-accelerated XTTS-v2 audio synthesis
- Multilingual text preprocessing
- Speaker voice cloning
- Whisper verification
- Duration filtering (skips segments < 0.5s)
- Automatic best GPU selection

**Usage:**
```bash
python generate_audio_with_tokenizer.py --language <lang> [--gpu-id GPU_ID]
```

### 5. **face_area_detector.py**
Analyzes face areas in video segments.

**Features:**
- GPU-accelerated OpenCV processing
- Face detection and area calculation
- Automatic best GPU selection
- Detailed visualization reports

**Usage:**
```bash
python face_area_detector.py <video_file> <output_dir> [--gpu-id GPU_ID]
```

## Support Modules

### **gpu_utils.py**
GPU management and optimization utilities.

**Features:**
- Automatic GPU detection and selection
- Best GPU selection based on free VRAM
- Multi-GPU parallel processing support
- GPU memory management and cache clearing
- Performance monitoring

### **multilingual_tokenizer.py**
Advanced text preprocessing for XTTS-v2 audio generation.

**Features:**
- **Multilingual Support**: English, Spanish, French, German, Portuguese, Italian, and more
- **Abbreviation Expansion**: "Mr." → "mister", "Sr." → "señor", "Dr." → "doctor"
- **Symbol Expansion**: "$" → "dollars/dólares", "%" → "percent/por ciento", "€" → "euros"
- **Number Expansion**: "14" → "fourteen/catorce", "20.50" → "twenty point five zero"
- **Smart Sentence Splitting**: Breaks long text into manageable chunks for TTS
- **Fallback Support**: Works with or without spaCy for sentence processing

**Supported Languages:**
- English (en)
- Spanish (es) 
- French (fr)
- German (de)
- Portuguese (pt)
- Italian (it)
- Polish (pl)
- Arabic (ar)
- Urdu (ur)
- Chinese (zh)
- Czech (cs)
- Russian (ru)
- Dutch (nl)
- Turkish (tr)
- Hungarian (hu)
- Korean (ko)
- Hindi (hi)

**Usage:**
```python
from multilingual_tokenizer import MultilingualTokenizer, tokenize_for_xtts

# Direct usage
processed_text = tokenize_for_xtts("Hello Mr. Smith! It costs $20.50.", language="en")

# Class usage
tokenizer = MultilingualTokenizer("es")
chunks = tokenizer.tokenize("Hola Sr. García. Cuesta $20.50.")
```

## GPU Acceleration Features

### Automatic GPU Selection
When no specific GPU is provided, all scripts automatically select the GPU with the most free VRAM:

```bash
# These commands will auto-select the best GPU:
python transcribe_segments.py
python translate_segments.py spanish
python generate_audio_with_tokenizer.py --language es
```

### GPU Memory Optimization
- **Smart Memory Detection**: Scans all GPUs and selects based on available VRAM
- **Cache Management**: Automatic GPU cache clearing after processing
- **Memory Monitoring**: Real-time VRAM usage tracking

### Multi-GPU Support
```bash
# Use specific GPU
python translate_segments.py spanish --gpu-id 1

# Enable parallel processing across multiple GPUs
python transcribe_segments.py --parallel
python translate_segments.py spanish --parallel
```

## Pipeline Configuration

### Complete Translation Pipeline
```bash
# 1. Speaker segmentation (auto-selects best GPU)
python speaker_audio_segmentation.py video1.mp4 speaker_segments

# 2. Transcription (auto-selects best GPU)  
python transcribe_segments.py

# 3. Translation to Spanish (auto-selects best GPU)
python translate_segments.py spanish

# 4. Audio generation with tokenizer (auto-selects best GPU)
python generate_audio_with_tokenizer.py --language es

# 5. Face area analysis (auto-selects best GPU)
python face_area_detector.py video1.mp4 face_area
```

### Multi-GPU Optimized Pipeline
```bash
# Distributed processing across 2 GPUs
python speaker_audio_segmentation.py video1.mp4 speaker_segments --gpu-id 0
python transcribe_segments.py --gpu-id 1 --parallel
python translate_segments.py spanish --gpu-id 0
python generate_audio_with_tokenizer.py --language es --gpu-id 1
python face_area_detector.py video1.mp4 face_area --gpu-id 0
```

## Performance Benefits

### GPU vs CPU Performance
- **Speaker Segmentation**: 10-15x faster on GPU
- **Whisper Transcription**: 5-8x faster on GPU
- **NLLB Translation**: 3-5x faster on GPU
- **XTTS Audio Generation**: 8-12x faster on GPU
- **Face Detection**: 20-30x faster on GPU

### Memory Requirements
- **Whisper (base)**: ~1GB VRAM
- **NLLB-200**: ~2-3GB VRAM  
- **XTTS-v2**: ~4-6GB VRAM
- **pyannote.audio**: ~1-2GB VRAM

## Troubleshooting

### Common Issues

**GPU Not Detected:**
```bash
# Check GPU availability
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**Out of Memory:**
- Use smaller batch sizes
- Clear GPU cache between operations
- Use CPU fallback: remove `--gpu-id` argument

**Text Processing Issues:**
- Install spaCy language models: `python -m spacy download en_core_web_sm`
- Install num2words: `pip install num2words`

**Audio Quality Issues:**
- Ensure proper speaker samples in `speaker_samples/`
- Use longer segments (>0.5s automatically filtered)
- Check Whisper verification scores in generation reports

### Dependencies

```bash
# Core ML libraries
pip install torch torchvision torchaudio
pip install transformers datasets
pip install whisper-openai
pip install TTS
pip install pyannote.audio

# Text processing
pip install num2words
pip install spacy

# Optional language models
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm
python -m spacy download fr_core_news_sm
```

## Output Structure

```
output_directory/
├── speaker_segments/          # Segmented audio files
├── transcribed_segments/      # Transcription results
├── translated_transcription/  # Translation results  
├── translated_audio/         # Generated audio files
├── face_area/               # Face analysis results
└── reports/                 # Processing reports
```

Each script generates detailed JSON reports with processing statistics, GPU usage information, and quality metrics. 