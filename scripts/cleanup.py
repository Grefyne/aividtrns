#!/usr/bin/env python3
"""
Cleanup script for audio processing pipeline.
Removes previous processing files and directories to start fresh.
"""

import os
import shutil
import argparse
import sys
from pathlib import Path


def cleanup_directories(directories, force=False):
    """Remove directories if they exist."""
    for directory in directories:
        if os.path.exists(directory):
            if force:
                print(f"Removing directory: {directory}")
                shutil.rmtree(directory)
            else:
                print(f"Directory exists (use --force to remove): {directory}")


def cleanup_files(files, force=False):
    """Remove files if they exist."""
    for file_path in files:
        if os.path.exists(file_path):
            if force:
                print(f"Removing file: {file_path}")
                os.remove(file_path)
            else:
                print(f"File exists (use --force to remove): {file_path}")


def main():
    parser = argparse.ArgumentParser(description="Cleanup previous processing files")
    parser.add_argument("--force", action="store_true", 
                       help="Force removal of files and directories")
    args = parser.parse_args()

    # Directories to clean up
    directories_to_clean = [
        "audio_export",
        "speaker_segments",
        "transcribed_segments", 
        "translated_transcription",
        "translated_segments",
        "translated_audio",
        "vocal_removal_out",
        "speaker_samples",
        "face_area",
        "extracted_faces",
        "latentsync_training_data",
        "temp_latentsync_input"
    ]

    # Files to clean up (keeping this for backward compatibility, but audio_export will be removed entirely)
    files_to_clean = [
        "audio_export/final_audio.wav",
        "audio_export/video1_audio_vocrem.wav"
    ]

    print("Audio Processing Pipeline Cleanup")
    print("=================================")

    if not args.force:
        print("Running in dry-run mode. Use --force to actually remove files.")
        print()

    # Clean up directories
    cleanup_directories(directories_to_clean, args.force)
    
    # Clean up specific files
    cleanup_files(files_to_clean, args.force)

    if args.force:
        print("\nCleanup completed successfully!")
    else:
        print("\nDry-run completed. Use --force to perform actual cleanup.")


if __name__ == "__main__":
    main() 