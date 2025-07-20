#!/usr/bin/env python3
"""
LatentSync training data preparation script.
Prepares training data for LatentSync v1.6 from extracted face areas and audio.
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path


def prepare_latentsync_training_data(input_dir, output_dir, num_workers=4, sequence_length=16):
    """Prepare LatentSync training data from extracted face areas and audio."""
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Preparing LatentSync training data from: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check input directory structure
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return False
    
    # Expected structure:
    # input_dir/
    #   SPEAKER_00/
    #   SPEAKER_01/
    #   audio/
    
    speaker_dirs = []
    audio_dir = None
    
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path):
            if item.startswith("SPEAKER_"):
                speaker_dirs.append(item)
            elif item == "audio":
                audio_dir = item_path
    
    if not speaker_dirs:
        print("Error: No speaker directories found in input directory")
        return False
    
    if not audio_dir:
        print("Warning: No audio directory found in input directory")
    
    print(f"Found {len(speaker_dirs)} speaker directories: {speaker_dirs}")
    
    # Create LatentSync training data structure
    training_data_structure = {
        "input_directory": input_dir,
        "output_directory": output_dir,
        "speakers": speaker_dirs,
        "audio_directory": audio_dir,
        "sequence_length": sequence_length,
        "num_workers": num_workers
    }
    
    # Copy face area videos to LatentSync format
    print("Copying face area videos to LatentSync format...")
    
    for speaker in speaker_dirs:
        speaker_input_dir = os.path.join(input_dir, speaker)
        speaker_output_dir = os.path.join(output_dir, speaker)
        
        os.makedirs(speaker_output_dir, exist_ok=True)
        
        # Look for face area video
        face_area_video = None
        for file in os.listdir(speaker_input_dir):
            if file.endswith("_face_area.mp4"):
                face_area_video = os.path.join(speaker_input_dir, file)
                break
        
        if face_area_video:
            # Copy to output directory
            output_video = os.path.join(speaker_output_dir, "face_area.mp4")
            shutil.copy2(face_area_video, output_video)
            print(f"  Copied {speaker} face area video")
        else:
            print(f"  Warning: No face area video found for {speaker}")
    
    # Copy audio files
    if audio_dir:
        print("Copying audio files...")
        audio_output_dir = os.path.join(output_dir, "audio")
        os.makedirs(audio_output_dir, exist_ok=True)
        
        for file in os.listdir(audio_dir):
            if file.endswith(('.wav', '.mp3', '.m4a')):
                src_path = os.path.join(audio_dir, file)
                dst_path = os.path.join(audio_output_dir, file)
                shutil.copy2(src_path, dst_path)
                print(f"  Copied audio file: {file}")
    
    # Create LatentSync configuration files
    create_latentsync_config(output_dir, training_data_structure)
    
    # Create training data summary
    create_training_data_summary(output_dir, training_data_structure)
    
    print(f"\nLatentSync training data preparation completed!")
    print(f"Training data saved to: {output_dir}")
    
    return True


def create_latentsync_config(output_dir, training_data_structure):
    """Create LatentSync configuration files."""
    
    print("Creating LatentSync configuration files...")
    
    # Create dataset configuration
    dataset_config = {
        "dataset_type": "face_audio_pairs",
        "sequence_length": training_data_structure["sequence_length"],
        "num_workers": training_data_structure["num_workers"],
        "speakers": training_data_structure["speakers"],
        "audio_directory": "audio" if training_data_structure["audio_directory"] else None,
        "face_video_suffix": "face_area.mp4",
        "audio_formats": [".wav", ".mp3", ".m4a"]
    }
    
    config_path = os.path.join(output_dir, "dataset_config.json")
    with open(config_path, 'w') as f:
        json.dump(dataset_config, f, indent=2)
    
    print(f"  Dataset config saved to: {config_path}")
    
    # Create training script template
    training_script = f"""#!/bin/bash
# LatentSync v1.6 Training Script
# Generated from audio processing pipeline

cd {os.path.join(os.path.dirname(output_dir), 'latentsync_integration')}

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Training parameters
SEQUENCE_LENGTH={training_data_structure["sequence_length"]}
NUM_WORKERS={training_data_structure["num_workers"]}
BATCH_SIZE=4
LEARNING_RATE=1e-4
NUM_EPOCHS=100

# Start training
python scripts/train_unet.py \\
    --config configs/unet/stage1.yaml \\
    --data_path {output_dir} \\
    --sequence_length $SEQUENCE_LENGTH \\
    --batch_size $BATCH_SIZE \\
    --learning_rate $LEARNING_RATE \\
    --num_epochs $NUM_EPOCHS \\
    --num_workers $NUM_WORKERS \\
    --checkpoint_dir checkpoints/unet \\
    --log_dir logs/unet
"""
    
    script_path = os.path.join(output_dir, "train_latentsync.sh")
    with open(script_path, 'w') as f:
        f.write(training_script)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    print(f"  Training script saved to: {script_path}")


def create_training_data_summary(output_dir, training_data_structure):
    """Create a summary of the prepared training data."""
    
    summary = {
        "dataset_info": {
            "name": "LatentSync Training Dataset",
            "version": "1.6",
            "description": "Training data prepared from audio processing pipeline",
            "creation_date": str(Path().cwd()),
            "source": training_data_structure["input_directory"]
        },
        "structure": {
            "output_directory": output_dir,
            "speakers": training_data_structure["speakers"],
            "has_audio": training_data_structure["audio_directory"] is not None,
            "sequence_length": training_data_structure["sequence_length"],
            "num_workers": training_data_structure["num_workers"]
        },
        "files": {
            "dataset_config": "dataset_config.json",
            "training_script": "train_latentsync.sh",
            "summary": "training_data_summary.json"
        }
    }
    
    # Count files in each speaker directory
    speaker_files = {}
    for speaker in training_data_structure["speakers"]:
        speaker_dir = os.path.join(output_dir, speaker)
        if os.path.exists(speaker_dir):
            files = os.listdir(speaker_dir)
            speaker_files[speaker] = {
                "files": files,
                "count": len(files)
            }
    
    summary["speaker_files"] = speaker_files
    
    # Count audio files
    audio_dir = os.path.join(output_dir, "audio")
    if os.path.exists(audio_dir):
        audio_files = os.listdir(audio_dir)
        summary["audio_files"] = {
            "files": audio_files,
            "count": len(audio_files)
        }
    
    # Save summary
    summary_path = os.path.join(output_dir, "training_data_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Training data summary saved to: {summary_path}")
    
    # Print summary
    print(f"\nTraining Data Summary:")
    print(f"  Speakers: {len(training_data_structure['speakers'])}")
    print(f"  Sequence length: {training_data_structure['sequence_length']}")
    print(f"  Has audio: {training_data_structure['audio_directory'] is not None}")
    
    for speaker, files_info in speaker_files.items():
        print(f"  {speaker}: {files_info['count']} files")
    
    if "audio_files" in summary:
        print(f"  Audio files: {summary['audio_files']['count']}")


def main():
    parser = argparse.ArgumentParser(description="Prepare LatentSync training data")
    parser.add_argument("--input_dir", required=True,
                       help="Input directory containing speaker face areas and audio")
    parser.add_argument("--output_dir", default="latentsync_training_data",
                       help="Output directory for LatentSync training data")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of workers for data loading")
    parser.add_argument("--sequence_length", type=int, default=16,
                       help="Sequence length for training")
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Prepare training data
    success = prepare_latentsync_training_data(
        args.input_dir, 
        args.output_dir, 
        args.num_workers, 
        args.sequence_length
    )
    
    if not success:
        print("Training data preparation failed!")
        sys.exit(1)
    else:
        print("LatentSync training data preparation completed successfully!")


if __name__ == "__main__":
    main() 