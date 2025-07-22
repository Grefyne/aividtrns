#!/usr/bin/env python3
"""
Compare durations of generated audio files with transcript durations for each segment.
Outputs a table: segment number | transcript duration | audio file duration | difference (s) | difference (%)
"""
import os
import json
import argparse
import subprocess

# Helper to get audio duration using ffprobe
def get_audio_duration(filepath):
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', filepath
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        return None

def main():
    parser = argparse.ArgumentParser(description="Compare transcript and audio durations for segments.")
    parser.add_argument('--transcript', default='transcribed_segments/transcribed_segments.json', help='Path to transcript JSON')
    parser.add_argument('--audio-dir', default='translated_audio', help='Directory with generated audio files')
    args = parser.parse_args()

    # Load transcript JSON
    with open(args.transcript, 'r', encoding='utf-8') as f:
        data = json.load(f)
    segments = data['segments']

    # Prepare table header
    header = ["Segment", "Transcript (s)", "Audio (s)", "Diff (s)", "Diff (%)"]
    rows = []

    for i, seg in enumerate(segments, 1):
        seg_id = seg['segment_id']
        speaker = seg['speaker']
        transcript_dur = float(seg['duration'])
        # Find audio file (match by segment_id and speaker)
        # The output audio file is usually named like: segmentXXX_SPEAKER_YY_...wav
        audio_file = None
        for fname in os.listdir(args.audio_dir):
            if fname.startswith(f"{seg_id}_{speaker}") and fname.endswith('.wav'):
                audio_file = os.path.join(args.audio_dir, fname)
                break
        if not audio_file or not os.path.exists(audio_file):
            audio_dur = None
        else:
            audio_dur = get_audio_duration(audio_file)
        # Compute differences
        if audio_dur is not None:
            diff = audio_dur - transcript_dur
            diff_pct = (diff / transcript_dur * 100) if transcript_dur else 0.0
            audio_dur_str = f"{audio_dur:.3f}"
            diff_str = f"{diff:+.3f}"
            diff_pct_str = f"{diff_pct:+.1f}%"
        else:
            audio_dur_str = "MISSING"
            diff_str = "-"
            diff_pct_str = "-"
        rows.append([
            f"{i:02d}",
            f"{transcript_dur:.3f}",
            audio_dur_str,
            diff_str,
            diff_pct_str
        ])

    # Print table
    col_widths = [max(len(str(row[i])) for row in [header]+rows) for i in range(len(header))]
    def print_row(row):
        print(" | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths)))
    print_row(header)
    print("-+-".join('-'*w for w in col_widths))
    for row in rows:
        print_row(row)

if __name__ == "__main__":
    main() 