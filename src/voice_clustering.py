# src/voice_clustering.py
from preprocess import preprocess_all
from embeddings import process_all
from cluster import cluster_embeddings

def run_pipeline():
    print("1) Preprocess (VAD & split)...")
    preprocess_all()
    print("2) Extract embeddings...")
    process_all()
    print("3) Cluster embeddings...")
    cluster_embeddings()

if __name__ == "__main__":
    run_pipeline()
