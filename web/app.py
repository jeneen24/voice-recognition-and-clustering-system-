# app.py
from flask import Flask, render_template, url_for
import pandas as pd
from pathlib import Path
import shutil
import os

app = Flask(__name__)
RESULTS = Path("data/results")
EMB = Path("data/embeddings")
CLEAN = Path("data/clean")
STATIC_AUDIO = Path("web/static/audio")
STATIC_AUDIO.mkdir(parents=True, exist_ok=True)

def prepare_static_audio():
    """Copy wav files referenced in clusters.csv into web/static/audio for serving."""
    csvp = RESULTS / "clusters.csv"
    if not csvp.exists():
        return {}
    df = pd.read_csv(csvp)
    clusters = {}
    for _, row in df.iterrows():
        fname = row['file']
        # find file under data/clean
        found = None
        for p in CLEAN.rglob(fname):
            found = p
            break
        if found is None:
            continue
        dest = STATIC_AUDIO / found.name
        if not dest.exists():
            shutil.copyfile(found, dest)
        relpath = f"audio/{found.name}"
        clusters.setdefault(row['cluster'], []).append({"file": found.name, "relpath": relpath})
    return clusters

@app.route("/")
def index():
    clusters = prepare_static_audio()
    cluster_counts = {str(c): len(items) for c, items in clusters.items()}
    return render_template("index.html", clusters=clusters, cluster_counts=cluster_counts)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
