# 🎙️ Voice Clustering Project (TASK6)

This project implements a **multi-speaker voice clustering pipeline**.  
It processes raw audio recordings, extracts meaningful speech segments, generates speaker embeddings, and groups them into clusters (potential speakers).  
A simple **web interface** (Flask + HTML/CSS/JS) is provided to visualize the results and listen to the clustered audio samples.

---

## 📂 Project Structure

TASK6/
├─ data/
│ ├─ raw/ # raw audio recordings (organized by speaker folders)
│ ├─ clean/ # VAD-split speech segments
│ ├─ embeddings/ # extracted speaker embeddings (.npy)
│ └─ results/ # clustering results (clusters.csv, plots, etc.)
├─ src/
│ ├─ preprocess.py # voice activity detection (VAD) and audio splitting
│ ├─ vad.py # VAD helper functions (webrtcvad)
│ ├─ embeddings.py # generate embeddings using Resemblyzer
│ ├─ cluster.py # clustering (Agglomerative / KMeans)
│ ├─ transcribe.py # (optional) speech-to-text using Whisper
│ └─ voice_clustering.py # full pipeline runner
├─ web/
│ ├─ static/
│ │ ├─ style.css
│ │ ├─ static.js
│ │ └─ audio/ # copied audio files for playback in UI
│ └─ templates/
│ └─ index.html
├─ app.py # Flask web server
├─ requirements.txt
└─ README.md


---

## 🚀 Features

1. **Preprocessing**
   - Splits raw audio into smaller speech segments using **WebRTC VAD**.
   - Ensures background noise and silence are removed.

2. **Speaker Embeddings**
   - Uses **Resemblyzer** to generate embeddings from each speech segment.
   - Stores embeddings as `.npy` files for clustering.

3. **Clustering**
   - Supports **Agglomerative Clustering** and **KMeans**.
   - Automatically estimates number of clusters if not provided.
   - Saves results into `data/results/clusters.csv`.

4. **Web Interface**
   - Displays all clusters with audio players for each segment.
   - Shows statistics (e.g., number of segments per cluster).
   - Interactive chart (using **Chart.js**) for cluster distribution.
   - "Refresh" button to reload updated clustering results.

5. **Extensibility**
   - Can integrate transcription (Whisper, SpeechBrain, PyAnnote).
   - Supports additional visualization (e.g., t-SNE/UMAP of embeddings).

---

## ⚙️ Installation

1. Clone this repository:
   


    Install Python dependencies:

pip install -r requirements.txt

Install system dependencies:

    sudo apt update
    sudo apt install ffmpeg sox

🎤 Usage
1. Prepare Data

Place your raw audio files inside:

data/raw/speaker1/
data/raw/speaker2/
...

2. Run Full Pipeline

python src/voice_clustering.py

This will:

    Split audio into clean speech segments

    Extract embeddings

    Perform clustering

    Save results into data/results/clusters.csv

3. Launch Web UI

python app.py

Open your browser at: http://127.0.0.1:5000
📊 Web Interface

    Cluster Overview: Each cluster is displayed with:

        Cluster ID

        Number of audio segments

        List of audio samples with playback controls

    Statistics:

        Bar chart showing distribution of segments across clusters

    Actions:

        "Refresh Results" button reloads clustering output without restarting server

🧪 Example Workflow

    Raw audio:

data/raw/jackson/1.wav
data/raw/lucas/2.wav
data/raw/theo/3.wav

Run pipeline:

python src/voice_clustering.py

Generated outputs:

    Segments → data/clean/jackson/jackson_seg0.wav ...

    Embeddings → data/embeddings/jackson_seg0.npy ...

    Results → data/results/clusters.csv

Web UI shows clusters, e.g.:

    Cluster 0 → {jackson_seg0.wav, lucas_seg1.wav}
    Cluster 1 → {theo_seg0.wav}



🛠️ Tech Stack

    Python: librosa, soundfile, webrtcvad, resemblyzer, scikit-learn, flask

    Frontend: HTML5, CSS3, JavaScript (Chart.js for charts)

    System: ffmpeg, sox