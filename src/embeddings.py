# src/embeddings.py
import os
from pathlib import Path
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
import soundfile as sf

CLEAN_DIR = Path("data/clean")
EMB_DIR = Path("data/embeddings")
EMB_DIR.mkdir(parents=True, exist_ok=True)

encoder = VoiceEncoder()

def extract_embedding(file_path):
    wav = preprocess_wav(file_path)  # returns samples in float32
    emb = encoder.embed_utterance(wav)
    return emb

def process_all():
    records = []
    for speaker_folder in CLEAN_DIR.iterdir():
        if speaker_folder.is_dir():
            for wav in speaker_folder.glob("*.wav"):
                try:
                    emb = extract_embedding(str(wav))
                    fname = EMB_DIR / (wav.stem + ".npy")
                    np.save(fname, emb)
                    records.append({"file": str(wav), "emb_path": str(fname)})
                    print("Saved embedding for", wav)
                except Exception as e:
                    print("Error embedding", wav, e)
    return records

if __name__ == "__main__":
    process_all()
