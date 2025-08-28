# src/preprocess.py
import os
from pathlib import Path
from vad import split_audio_vad

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")

def preprocess_all(aggressiveness=2):
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    for speaker_folder in RAW_DIR.iterdir():
        if speaker_folder.is_dir():
            out_speaker_dir = CLEAN_DIR / speaker_folder.name
            out_speaker_dir.mkdir(parents=True, exist_ok=True)
            for file in speaker_folder.glob("*"):
                if file.suffix.lower() in [".wav", ".flac", ".mp3", ".m4a"]:
                    try:
                        print(f"Processing {file} ...")
                        segs = split_audio_vad(str(file), str(out_speaker_dir), aggressiveness=aggressiveness)
                        print(f"  -> {len(segs)} segments")
                    except Exception as e:
                        print("Error processing", file, e)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggr", type=int, default=2, help="VAD aggressiveness 0-3")
    args = parser.parse_args()
    preprocess_all(aggressiveness=args.aggr)
