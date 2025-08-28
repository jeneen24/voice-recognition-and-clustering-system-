# src/transcribe.py
import os
from pathlib import Path
# import whisper

CLEAN_DIR = Path("data/clean")
RESULTS_DIR = Path("data/results")

def transcribe_all(model_name="base"):
    print("Transcription disabled by default. Uncomment whisper code to enable.")
    # model = whisper.load_model(model_name)
    # for speaker in CLEAN_DIR.iterdir():
    #     for wav in speaker.glob("*.wav"):
    #         res = model.transcribe(str(wav))
    #         out = RESULTS_DIR / (wav.stem + ".txt")
    #         out.write_text(res["text"], encoding="utf-8")
    #         print("Transcribed", wav)

if __name__ == "__main__":
    transcribe_all()
