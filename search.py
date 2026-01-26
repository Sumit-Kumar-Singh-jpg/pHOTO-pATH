# search.py
import os, sys, json, time
import numpy as np
import faiss
import torch
import clip
from PIL import Image

# ──────────────────────────────
# CONFIG
# ──────────────────────────────
META_DIR = "metadata"
EMBED_FILE = os.path.join(META_DIR, "embeddings.npy")
META_FILE = os.path.join(META_DIR, "metadata.json")
MODEL_NAME = "ViT-B/32"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

# ──────────────────────────────
# LOAD INDEX + MODEL
# ──────────────────────────────
embeddings = np.load(EMBED_FILE)
with open(META_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

dim = embeddings.shape[1]
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized inner product
index.add(embeddings)

model, preprocess = clip.load(MODEL_NAME, device=device)

# ──────────────────────────────
# SEARCH FUNCTIONS
# ──────────────────────────────
def text_search(query, topk=5):
    tokens = clip.tokenize([query]).to(device)
    with torch.no_grad():
        tvec = model.encode_text(tokens)
        tvec = tvec / tvec.norm(dim=-1, keepdim=True)
    tvec = tvec.cpu().numpy().astype("float32")

    faiss.normalize_L2(tvec)
    D, I = index.search(tvec, topk)
    results = [
    (metadata[i].get("filename") or metadata[i].get("path"), float(D[0][k]))
    for k, i in enumerate(I[0])
]
    return results


def image_search(image_path, topk=5):
    img = Image.open(image_path).convert("RGB")
    img_t = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        ivec = model.encode_image(img_t)
        ivec = ivec / ivec.norm(dim=-1, keepdim=True)
    ivec = ivec.cpu().numpy().astype("float32")

    faiss.normalize_L2(ivec)
    D, I = index.search(ivec, topk)
    results = [
    (metadata[i].get("filename") or metadata[i].get("path"), float(D[0][k]))
    for k, i in enumerate(I[0])
]
    return results

# ──────────────────────────────
# MAIN ENTRY
# ──────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python search.py text \"sunset beach\" 5")
        print("  python search.py image path/to/query.jpg 5")
        sys.exit(1)

    mode = sys.argv[1].lower()
    query = sys.argv[2]
    topk = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    start = time.time()
    if mode == "text":
        results = text_search(query, topk)
        print(f"\nTop {topk} results for query: '{query}'\n")
    elif mode == "image":
        results = image_search(query, topk)
        print(f"\nTop {topk} visually similar results to: {query}\n")
    else:
        print("Error: mode must be 'text' or 'image'")
        sys.exit(1)

    for rank, (fn, score) in enumerate(results, 1):
        print(f"[{rank}] {score:.4f}  {fn}")

    print(f"\n⏱ Search completed in {time.time() - start:.3f}s")
