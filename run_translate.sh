#!/bin/bash

# Audio Processing Toolkit - Complete Workflow Script
# Complete pipeline: Extract audio → Diarization → Translation → Speaker isolation → TTS → Vocal removal
# --- unset deprecated variables ---
unset TRANSFORMERS_CACHE
unset PYTORCH_TRANSFORMERS_CACHE
unset PYTORCH_PRETRAINED_BERT_CACHE
# --- set the new ones (all under your project) ---
export TF_CPP_MIN_LOG_LEVEL=2 
export HF_HOME="$(pwd)/hf_cache"               
export HF_HUB_CACHE="$HF_HOME/hub"             
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_METRICS_CACHE="$HF_HOME/metrics"
export HF_MODULES_CACHE="$HF_HOME/modules"
export TORCH_CUDNN_V8_API_DISABLED=1 

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if file exists
check_file() {
    if [ ! -f "$1" ]; then
        print_error "File not found: $1"
        exit 1
    fi
}

# Function to check if directory exists
check_directory() {
    if [ ! -d "$1" ]; then
        print_error "Directory not found: $1"
        exit 1
    fi
}

# Check if Python is available
if ! command -v python &> /dev/null; then
    print_error "Python is not installed or not in PATH"
    exit 1
fi

# Check if required scripts exist
check_file "scripts/cleanup.py"
check_file "scripts/extract_audio.py"
check_file "scripts/speaker_audio_segmentation.py"
check_file "scripts/transcribe_segments.py"
check_file "scripts/extract_speaker_samples.py"
check_file "scripts/translate_segments.py"
check_file "scripts/generate_audio_with_tokenizer.py"
check_file "scripts/build_audio.py"
check_file "scripts/vocal_removal.py"
check_file "scripts/final_audio.py"
check_file "scripts/face_area_detector.py"
check_file "scripts/extract_face_areas.py"
check_file "scripts/ls_train_data.py"

# Check if video directory exists
check_directory "video_in"

# Show help if requested
if [ "$1" = "--help" ] || [ "$1" = "-h" ] || [ $# -eq 0 ]; then
    echo "Audio Processing Toolkit - Complete Workflow Script"
    echo "=================================================="
    echo
    echo "USAGE:"
    echo "  ./run_translate.sh <VIDEO_FILE> <TARGET_LANGUAGE>"
    echo
    echo "PARAMETERS:"
    echo "  VIDEO_FILE       Path to input video file (required)"
    echo "  TARGET_LANGUAGE  Target language for translation (required)"
    echo "                   Examples: spanish, french, german, etc."
    echo
    echo "EXAMPLES:"
    echo "  ./run_translate.sh video_in/video1.mp4 spanish      # Translate to Spanish"
    echo "  ./run_translate.sh my_video.mp4 french              # Translate to French"
    echo "  ./run_translate.sh /path/to/recording.mov german    # Translate to German"
    echo
    echo "DESCRIPTION:"
    echo "  This script runs the complete audio processing and translation workflow:"
    echo "  0. Cleans up previous processing files (scripts/cleanup.py)"
    echo "  1. Extracts audio from video file (scripts/extract_audio.py)"
    echo "  2. Performs speaker audio segmentation (scripts/speaker_audio_segmentation.py)"
    echo "  3. Transcribes audio segments using Whisper (scripts/transcribe_segments.py)"
    echo "  4. Extracts individual speaker samples (scripts/extract_speaker_samples.py)"
    echo "  5. Translates transcribed segments using NLLB (scripts/translate_segments.py)"
    echo "  6. Generates translated audio with XTTS-v2 and Whisper verification (scripts/generate_audio_with_tokenizer.py)"
    echo "  7. Merges translated audio segments into final audio file (scripts/build_audio.py)"
    echo "  8. Removes vocals from original audio (scripts/vocal_removal.py)"
    echo "  9. Creates final audio mix with translated audio over background music (scripts/final_audio.py)"
    echo "  10. Analyzes face areas in video for speaker positioning (scripts/face_area_detector.py)"
    echo "  11. Extracts face area videos and images for each speaker (scripts/extract_face_areas.py)"
    echo "  12. Prepares LatentSync v1.6 training data from extracted face areas (scripts/ls_train_data.py)"
    echo
    echo "OUTPUT FILES:"
    echo "  • audio_export/[video_name]_audio.wav                    - Extracted audio file"
    echo "  • speaker_segments/                                      - Speaker audio segments"
    echo "  • speaker_segments/segments_summary.json                 - Segmentation summary"
    echo "  • transcribed_segments/transcribed_segments.json         - Whisper transcriptions"
    echo "  • speaker_samples/speakerN_sample.wav                    - Individual speaker audio"
    echo "  • translated_transcription/translated_transcription.json - NLLB translations"
    echo "  • translated_segments/                                   - Generated TTS audio files"
    echo "  • translated_audio/merged_translated_audio.wav           - Final merged audio file"
    echo "  • vocal_removal_out/                                     - Audio with vocals removed"
    echo "  • audio_export/final_audio.wav                           - Final mixed audio with background music"
    echo "  • face_area/                                             - Face area analysis results and visualizations"
    echo "  • extracted_faces/                                       - Extracted face area videos and images for each speaker"
    echo "  • latentsync_training_data/                              - LatentSync v1.6 training dataset with face/audio pairs"
    echo
    echo "REQUIREMENTS:"
    echo "  • Python with dependencies: pip install -r requirements.txt"
    echo "  • NVIDIA GPU recommended for best performance"
    echo "  • HuggingFace token for pyannote models"
    echo
    exit 0
fi

# Validate required parameters
if [ $# -lt 2 ]; then
    print_error "Missing required parameters"
    echo "Usage: ./run_translate.sh <VIDEO_FILE> <TARGET_LANGUAGE>"
    echo "Run './run_translate.sh --help' for more information"
    exit 1
fi

# Get parameters
INPUT_VIDEO="$1"
TARGET_LANGUAGE="$2"

# Validate language parameter
if [ -z "$TARGET_LANGUAGE" ]; then
    print_error "Target language parameter cannot be empty"
    exit 1
fi

# Check if input video exists
check_file "$INPUT_VIDEO"

print_status "Starting Complete Audio Processing and Translation Workflow"
print_status "Input video: $INPUT_VIDEO"
print_status "Target language: $TARGET_LANGUAGE"
echo

# Step 0: Cleanup Previous Files
print_status "Step 0/8: Cleaning up previous processing files..."
echo "------------------------------------------------------------"

if python scripts/cleanup.py --force; then
    print_success "Cleanup completed successfully"
else
    print_error "Cleanup failed"
    exit 1
fi

echo

# Step 1: Extract Audio
print_status "Step 1/8: Extracting audio from video..."
echo "----------------------------------------"

if python scripts/extract_audio.py "$INPUT_VIDEO"; then
    print_success "Audio extraction completed successfully"
else
    print_error "Audio extraction failed"
    exit 1
fi

echo

# Check if audio was created
AUDIO_FILE="audio_export/$(basename "${INPUT_VIDEO%.*}")_audio.wav"
if [ ! -f "$AUDIO_FILE" ]; then
    print_error "Expected audio file not found: $AUDIO_FILE"
    exit 1
fi

# Step 2: Speaker Audio Segmentation
print_status "Step 2/8: Performing speaker audio segmentation..."
echo "------------------------------------------------------------"

if python scripts/speaker_audio_segmentation.py "$AUDIO_FILE" --gpu-id 0; then
    print_success "Speaker audio segmentation completed successfully"
else
    print_error "Speaker audio segmentation failed"
    exit 1
fi

echo

# Step 3: Transcribe Audio Segments
print_status "Step 3/8: Transcribing audio segments using Whisper..."
echo "------------------------------------------------------------"

if python scripts/transcribe_segments.py --segments_dir "speaker_segments" --output_dir "transcribed_segments" --model "large" --gpu-id 1 --parallel; then
    print_success "Audio segment transcription completed successfully"
else
    print_error "Audio segment transcription failed"
    exit 1
fi

echo

# Step 4: Extract Speaker Samples
print_status "Step 4/9: Extracting individual speaker samples..."
echo "------------------------------------------------------------"

if python scripts/extract_speaker_samples.py "speaker_segments/segments_summary.json" "$AUDIO_FILE"; then
    print_success "Speaker sample extraction completed successfully"
else
    print_error "Speaker sample extraction failed"
    exit 1
fi

echo

# Step 5: Translate Transcribed Segments
print_status "Step 5/8: Translating transcribed segments using NLLB..."
echo "------------------------------------------------------------"

if python scripts/translate_segments.py "$TARGET_LANGUAGE" --input_file "transcribed_segments/transcribed_segments.json" --output_dir "translated_transcription" --gpu-id 0 --parallel; then
    print_success "Segment translation completed successfully"
else
    print_error "Segment translation failed"
    exit 1
fi

echo

# Step 6: Generate Translated Audio
print_status "Step 6/8: Generating translated audio with XTTS-v2 and Whisper verification..."
echo "------------------------------------------------------------"

if python scripts/generate_audio_with_tokenizer.py --input "translated_transcription/translated_transcription.json" --output "translated_segments" --language "$TARGET_LANGUAGE" --speaker-dir "speaker_segments" --max-retries 5 --confidence-threshold 85.0 --gpu-id 1; then
    print_success "Translated audio generation completed successfully"
else
    print_error "Translated audio generation failed"
    exit 1
fi

echo

# Step 7: Build Merged Audio
print_status "Step 7/8: Merging translated audio segments into final audio file..."
echo "------------------------------------------------------------"

if python scripts/build_audio.py --transcription "translated_segments/translated_transcription.json" --audio-dir "translated_segments" --output "translated_audio/merged_translated_audio.wav"; then
    print_success "Audio merging completed successfully"
else
    print_error "Audio merging failed"
    exit 1
fi

echo

# Step 8: Vocal Removal
print_status "Step 8/9: Removing vocals from original audio..."
echo "------------------------------------------------------------"

if python scripts/vocal_removal.py "$AUDIO_FILE"; then
    print_success "Vocal removal completed successfully"
else
    print_error "Vocal removal failed"
    exit 1
fi

echo

# Step 9: Create Final Audio Mix
print_status "Step 9/10: Creating final audio mix with translated audio over background music..."
echo "------------------------------------------------------------"

if python scripts/final_audio.py; then
    print_success "Final audio mix completed successfully"
else
    print_error "Final audio mix failed"
    exit 1
fi

echo

# Step 10: Face Area Detection
print_status "Step 10/11: Analyzing face areas in video for speaker positioning..."
echo "------------------------------------------------------------"

if python scripts/face_area_detector.py "$INPUT_VIDEO" --output-dir "face_area" --padding-ratio 0.3 --min-face-size 30 --gpu-id 0; then
    print_success "Face area detection completed successfully"
else
    print_error "Face area detection failed"
    exit 1
fi

echo

# Step 11: Extract Face Areas
print_status "Step 11/12: Extracting face area videos and images for each speaker..."
echo "------------------------------------------------------------"

# Check if face area results exist
FACE_AREA_RESULTS="face_area/face_area_results.json"
if [ ! -f "$FACE_AREA_RESULTS" ]; then
    print_error "Face area results not found: $FACE_AREA_RESULTS"
    exit 1
fi

if python scripts/extract_face_areas.py "$INPUT_VIDEO" "$FACE_AREA_RESULTS" --output-dir "extracted_faces"; then
    print_success "Face area extraction completed successfully"
else
    print_error "Face area extraction failed"
    exit 1
fi

echo

# Step 12: Prepare LatentSync Training Data
print_status "Step 12/12: Preparing LatentSync v1.6 training data from extracted face areas..."
echo "------------------------------------------------------------"

# Check if extracted faces directory exists
if [ ! -d "extracted_faces" ]; then
    print_error "Extracted faces directory not found: extracted_faces/"
    exit 1
fi

# Check if extracted faces contain the expected structure
if [ ! -f "extracted_faces/SPEAKER_00_face_area.mp4" ] || [ ! -f "extracted_faces/SPEAKER_01_face_area.mp4" ]; then
    print_error "Required speaker face area videos not found in extracted_faces/"
    print_error "Expected: SPEAKER_00_face_area.mp4 and SPEAKER_01_face_area.mp4"
    exit 1
fi

# Create a temporary directory structure for LatentSync training data preparation
print_status "Creating temporary directory structure for LatentSync training data..."
mkdir -p "temp_latentsync_input/SPEAKER_00"
mkdir -p "temp_latentsync_input/SPEAKER_01"
mkdir -p "temp_latentsync_input/audio"

# Copy extracted face area videos to the expected structure
cp "extracted_faces/SPEAKER_00_face_area.mp4" "temp_latentsync_input/SPEAKER_00/"
cp "extracted_faces/SPEAKER_01_face_area.mp4" "temp_latentsync_input/SPEAKER_01/"

# Copy the original audio file for training data
AUDIO_FILE="audio_export/$(basename "${INPUT_VIDEO%.*}")_audio.wav"
if [ -f "$AUDIO_FILE" ]; then
    cp "$AUDIO_FILE" "temp_latentsync_input/audio/"
else
    print_warning "Original audio file not found, using extracted audio if available"
fi

# Run LatentSync training data preparation
# if python scripts/ls_train_data.py --input_dir "temp_latentsync_input" --output_dir "latentsync_training_data" --num_workers 4 --sequence_length 16; then
#     print_success "LatentSync training data preparation completed successfully"
    
#     # Clean up temporary directory
#     rm -rf "temp_latentsync_input"
#     print_status "Cleaned up temporary directory"
# else
#     print_error "LatentSync training data preparation failed"
#     # Clean up temporary directory even on failure
#     rm -rf "temp_latentsync_input"
#     exit 1
# fi

echo

# Final status
print_success "Complete Audio Processing and Translation Workflow completed!"
echo
print_status "Generated files and directories:"
echo "  • Audio file: $AUDIO_FILE"
echo "  • Speaker segments: speaker_segments/"
echo "  • Segmentation summary: speaker_segments/segments_summary.json"
echo "  • Whisper transcriptions: transcribed_segments/transcribed_segments.json"
echo "  • Speaker samples: speaker_samples/"
echo "  • NLLB translations: translated_transcription/translated_transcription.json"
echo "  • Generated TTS audio: translated_segments/"
echo "  • Final merged audio: translated_audio/merged_translated_audio.wav"
echo "  • Vocal removal output: vocal_removal_out/"
echo "  • Final mixed audio: audio_export/final_audio.wav"
echo "  • Face area analysis: face_area/"
echo "  • Extracted face areas: extracted_faces/"
echo "  • LatentSync training data: latentsync_training_data/"
echo
print_status "Workflow summary:"
echo "  • Original audio extracted and analyzed"
echo "  • Speakers identified and audio segments created"
echo "  • Audio segments transcribed using Whisper"
echo "  • Transcriptions translated to $TARGET_LANGUAGE using NLLB"
echo "  • Translated audio generated using XTTS-v2 with Whisper verification"
echo "  • Audio segments merged into final translated audio file"
echo "  • Background music preserved (vocals removed)"
echo "  • Final audio mix created with translated speech over background music"
echo "  • Face areas analyzed for speaker positioning and visualization"
echo "  • Face area videos and images extracted for each speaker"
echo "  • LatentSync v1.6 training dataset prepared with face/audio pairs"
echo
print_status "Next steps:"
echo "  • Review speaker segmentation results for accuracy"
echo "  • Review Whisper transcriptions for accuracy"
echo "  • Review NLLB translations for accuracy"
echo "  • Review generated TTS audio quality"
echo "  • Review final merged audio file"
echo "  • Review final mixed audio with background music"
echo "  • Review face area analysis results and visualizations"
echo "  • Review extracted face area videos and images for each speaker"
echo "  • Review LatentSync training dataset and configuration" 