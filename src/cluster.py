# src/cluster.py
import os
from pathlib import Path
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
import pandas as pd

EMB_DIR = Path("data/embeddings")
RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_embeddings():
    rows = []
    for f in EMB_DIR.glob("*.npy"):
        emb = np.load(f)
        rows.append({"emb_path": str(f), "file": f.stem + ".wav", "emb": emb})
    return rows

def cluster_embeddings(method="agg", n_clusters=None):
    rows = load_embeddings()
    if not rows:
        print("No embeddings found.")
        return None
    X = np.vstack([r["emb"] for r in rows])
    if n_clusters is None:
        # heuristic: sqrt(#samples)
        n_clusters = max(2, int(len(rows)**0.5))
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    for r, lbl in zip(rows, labels):
        r["cluster"] = int(lbl)
    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "clusters.csv"
    df2 = df[["file", "emb_path", "cluster"]]
    df2.to_csv(out_csv, index=False)
    print("Saved clusters to", out_csv)
    summary = df2.groupby("cluster").size().reset_index(name="count")
    print("Cluster summary:\n", summary)
    return df2

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="agg")
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()
    cluster_embeddings(method=args.method, n_clusters=args.k)
