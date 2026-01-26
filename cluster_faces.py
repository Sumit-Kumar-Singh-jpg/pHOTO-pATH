# cluster_faces.py
import argparse, os, json
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity

META_DIR = "metadata"
FACE_EMBED_FILE = os.path.join(META_DIR, "faces.npy")
FACE_META_FILE  = os.path.join(META_DIR, "face_meta.json")
CLUSTER_STATS   = os.path.join(META_DIR, "cluster_stats.json")

def load_data():
    if not os.path.exists(FACE_EMBED_FILE):
        raise FileNotFoundError("⚠️ Missing faces.npy — run face_indexer.py first")

    if not os.path.exists(FACE_META_FILE):
        raise FileNotFoundError("⚠️ Missing face_meta.json — run face_indexer.py first")

    X = np.load(FACE_EMBED_FILE).astype(np.float32)
    with open(FACE_META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if len(meta) != len(X):
        raise RuntimeError(
            f"❌ faces.npy count ({len(X)}) != face_meta.json entries ({len(meta)})\n"
            f"Run: python face_indexer.py --rebuild\n"
        )

    # L2 normalize for cosine similarity
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X = X / norms
    return X, meta

def choose_k(X):
    n = len(X)
    if n < 20: return max(1, n // 2)
    if n < 80: return 6
    if n < 200: return 8
    return min(18, max(8, n // 25))  # scale smoothly

def cluster_dbscan(X, eps=0.21, min_samples=2):
    model = DBSCAN(metric="cosine", eps=eps, min_samples=min_samples)
    return model.fit_predict(X)

def cluster_kmeans(X, k):
    model = KMeans(n_clusters=k, n_init="auto", random_state=42)
    return model.fit_predict(X)

def pick_representatives(X, labels):
    reps = {}
    for c in sorted(set(labels)):
        if c == -1: continue  # noise
        idxs = np.where(labels == c)[0]
        if len(idxs) == 1:
            reps[c] = int(idxs[0]); continue
        
        centroid = X[idxs].mean(axis=0, keepdims=True)
        centroid /= (np.linalg.norm(centroid) + 1e-12)
        sims = cosine_similarity(centroid, X[idxs])[0]
        reps[c] = int(idxs[np.argmax(sims)])
    return reps

def main():
    p = argparse.ArgumentParser(description="Cluster face embeddings")
    p.add_argument("--algo", choices=["dbscan","kmeans"], default="dbscan")
    p.add_argument("--eps", type=float, default=0.21)
    p.add_argument("--min-samples", type=int, default=2)
    p.add_argument("--k", type=int, default=0)
    args = p.parse_args()

    X, meta = load_data()

    if args.algo == "dbscan":
        labels = cluster_dbscan(X, eps=args.eps, min_samples=args.min_samples)
        algo_info = {"algo":"dbscan", "eps":args.eps, "min_samples":args.min_samples}
    else:
        k = args.k if args.k > 0 else choose_k(X)
        labels = cluster_kmeans(X, k=k)
        algo_info = {"algo":"kmeans", "k":k}

    reps = pick_representatives(X, labels)

    noise = int(np.sum(labels == -1))
    clusters = sorted([c for c in set(labels) if c != -1])

    # Write cluster labels into metadata
    for i, m in enumerate(meta):
        m["cluster"] = int(labels[i])

    with open(FACE_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    stats = {
        "total": len(labels),
        "n_clusters": len(clusters),
        "clusters": [
            {"id": int(c), "size": int(np.sum(labels == c)), "rep_idx": int(reps.get(c,-1))}
            for c in clusters
        ],
        "noise": noise,
        "algo": algo_info,
    }
    with open(CLUSTER_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("✅ Clustering complete")
    print(f"   Faces: {len(labels)}")
    print(f"   Clusters: {len(clusters)}")
    print(f"   Noise faces: {noise}")
    print(f"   Stats saved → {CLUSTER_STATS}")

if __name__ == "__main__":
    main()
