#!/usr/bin/env python3
"""
train_xtts.py
Fine-tune XTTS-v2 for one speaker.
Run once per speaker; each run keeps the base model untouched.
"""

import os, argparse
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.xtts import Xtts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, help="Folder with wavs/ + metadata.csv")
    parser.add_argument("--out_dir", required=True, help="Where to save the checkpoint")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    # 1.  Load base config
    config = XttsConfig()
    config.load_json("XTTS-v2/config.json")  # downloaded from HF

    # 2.  Override training settings
    config.output_path           = args.out_dir
    config.datasets              = [{
        "name": "dataset",
        "path": args.dataset_dir,
        "meta_file_train": "metadata.csv",
        "language": "es"               # 'es', 'en', 'fr