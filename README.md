# Audio Processing and Translation Pipeline

A comprehensive audio processing toolkit that extracts audio from videos, performs speaker diarization, transcribes speech, translates content, and generates translated audio with voice cloning capabilities.

## 🎯 Features

- **Audio Extraction**: Extract high-quality audio from video files
- **Speaker Diarization**: Identify and separate different speakers using pyannote.audio
- **Speech Transcription**: Transcribe audio using OpenAI's Whisper model
- **Multi-language Translation**: Translate transcriptions using Facebook's NLLB-200 model
- **Voice Cloning**: Generate translated audio using XTTS-v2 with speaker voice cloning
- **Face Analysis**: Detect and extract face areas for speaker visualization
- **LatentSync Training**: Prepare training data for lip-sync models

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- NVIDIA GPU (recommended for best performance)
- FFmpeg installed
- HuggingFace account with access to pyannote.audio models

### Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd aividtrns
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up HuggingFace token**:
   ```bash
   mkdir config
   echo "your_huggingface_token_here" > config/hf_token.txt
   ```

4. **Place your video file**:
   ```bash
   mkdir video_in
   # Copy your video file to video_in/
   ```

### Usage

#### Complete Pipeline

Run the entire pipeline with one command:

```bash
./run_translate.sh video_in/your_video.mp4 spanish
```

This will:
1. Extract audio from the video
2. Perform speaker diarization
3. Transcribe audio segments
4. Extract speaker voice samples
5. Translate transcriptions to Spanish
6. Generate translated audio with voice cloning
7. Create final mixed audio with background music
8. Analyze face areas for speaker visualization

#### Individual Scripts

You can also run individual components:

```bash
# Extract audio
python scripts/extract_audio.py video_in/your_video.mp4

# Speaker segmentation
python scripts/speaker_audio_segmentation.py audio_export/video_audio.wav

# Transcribe segments
python scripts/transcribe_segments.py

# Extract speaker samples
python scripts/extract_speaker_samples.py speaker_segments/segments_summary.json audio_export/video_audio.wav

# Translate segments
python scripts/translate_segments.py spanish

# Generate translated audio
python scripts/generate_audio_with_tokenizer.py --input translated_transcription/translated_transcription.json --output translated_segments --language spanish

# Build final audio
python scripts/build_audio.py --transcription translated_segments/translated_transcription.json --audio-dir translated_segments --output translated_audio/merged_translated_audio.wav
```

## 📁 Project Structure

```
aividtrns/
├── scripts/                    # Main processing scripts
│   ├── extract_audio.py       # Audio extraction from video
│   ├── speaker_audio_segmentation.py  # Speaker diarization
│   ├── transcribe_segments.py # Speech transcription
│   ├── extract_speaker_samples.py # Voice sample extraction
│   ├── translate_segments.py  # Text translation
│   ├── generate_audio_with_tokenizer.py # TTS with voice cloning
│   ├── build_audio.py         # Audio merging
│   ├── vocal_removal.py       # Vocal removal
│   ├── final_audio.py         # Final audio mixing
│   ├── face_area_detector.py  # Face detection
│   ├── extract_face_areas.py  # Face area extraction
│   ├── ls_train_data.py       # LatentSync training data
│   └── cleanup.py             # Cleanup utility
├── video_in/                  # Input video files
├── config/                    # Configuration files
│   └── hf_token.txt          # HuggingFace token
├── run_translate.sh          # Complete pipeline script
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🔧 Configuration

### Supported Languages

The translation pipeline supports 200+ languages through NLLB-200:

- **European**: Spanish, French, German, Italian, Portuguese, Dutch, Swedish, Norwegian, Danish, Finnish, Polish, Czech, Hungarian, Romanian, Bulgarian, Greek, Turkish
- **Asian**: Chinese, Japanese, Korean, Thai, Vietnamese, Indonesian, Malay, Filipino
- **Middle Eastern**: Arabic, Hebrew, Persian, Turkish
- **African**: Swahili, Yoruba, Igbo, Hausa, Amharic, Somali, Zulu, Xhosa, Afrikaans
- **And many more...**

### Model Specifications

- **Speaker Diarization**: pyannote/speaker-diarization-3.1
- **Speech Recognition**: OpenAI Whisper (large model)
- **Translation**: Facebook NLLB-200 (distilled 600M)
- **Text-to-Speech**: XTTS-v2 with voice cloning
- **Face Detection**: OpenCV with dlib

## 📊 Output Files

The pipeline generates several output directories:

- `audio_export/` - Extracted audio files
- `speaker_segments/` - Individual speaker audio segments
- `transcribed_segments/` - Whisper transcriptions
- `speaker_samples/` - Speaker voice samples for cloning
- `translated_transcription/` - NLLB translations
- `translated_segments/` - Generated TTS audio files
- `translated_audio/` - Final merged translated audio
- `vocal_removal_out/` - Audio with vocals removed
- `face_area/` - Face area analysis results
- `extracted_faces/` - Extracted face area videos/images
- `latentsync_training_data/` - Training data for lip-sync models

## 🎵 Audio Quality

- **Input**: Any video format supported by FFmpeg
- **Audio Extraction**: 16kHz mono WAV (optimal for speech processing)
- **Speaker Segments**: Individual WAV files per speaker turn
- **Final Output**: High-quality translated audio with preserved background music

## 🔍 Speaker Segmentation

The improved speaker segmentation algorithm:
- Merges consecutive segments from the same speaker
- Reduces segment count by ~60% while maintaining quality
- Provides better context for transcription and translation
- Uses 1-second gap tolerance for natural conversation flow

## 🎭 Voice Cloning

The voice cloning system:
- Extracts high-quality voice samples from each speaker
- Uses XTTS-v2 for natural-sounding translated speech
- Maintains speaker-specific voice characteristics
- Verifies audio quality with Whisper

## 🖼️ Face Analysis

The face analysis system:
- Detects and clusters face areas in video frames
- Extracts face area videos for each speaker
- Prepares data for LatentSync lip-sync training
- Creates visualizations of speaker positioning

## 🛠️ Troubleshooting

### Common Issues

1. **HuggingFace Token Error**:
   - Ensure your token is valid and has access to pyannote.audio models
   - Check that `config/hf_token.txt` exists and contains your token

2. **CUDA/GPU Issues**:
   - Install compatible PyTorch version for your CUDA version
   - Check GPU memory availability for large models

3. **FFmpeg Not Found**:
   - Install FFmpeg: `sudo apt install ffmpeg` (Ubuntu/Debian)
   - Or: `brew install ffmpeg` (macOS)

4. **Model Download Issues**:
   - Check internet connection
   - Clear HuggingFace cache if needed: `rm -rf ~/.cache/huggingface/`

### Performance Tips

- Use GPU for faster processing
- Increase batch sizes if memory allows
- Use smaller Whisper models for faster transcription
- Process videos in smaller chunks for very long content

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the script documentation
3. Open an issue on GitHub

## 🔄 Version History

- **v1.0**: Initial release with basic pipeline
- **v1.1**: Added improved speaker segmentation
- **v1.2**: Enhanced voice cloning with XTTS-v2
- **v1.3**: Added face analysis and LatentSync support

---

**Note**: This project requires significant computational resources and is designed for research and development purposes. Ensure you have proper licenses for any commercial use of the underlying models. 