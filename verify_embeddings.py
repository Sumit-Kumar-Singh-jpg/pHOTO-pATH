import numpy as np, json

emb = np.load("metadata/embeddings.npy")
print("Shape:", emb.shape)
print("Dtype:", emb.dtype)
print("NaN count:", np.isnan(emb).sum())
print("Inf count:", np.isinf(emb).sum())
print("Min:", emb.min(), "Max:", emb.max())

with open("metadata/metadata.json") as f:
    meta = json.load(f)
print("Metadata entries:", len(meta))

if emb.shape[0] != len(meta):
    print("❌ Mismatch between embeddings and metadata!")
else:
    print("✅ Count matches.")
