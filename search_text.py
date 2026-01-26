import numpy as np, torch, clip, json

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load stored embeddings and metadata
embs = np.load("metadata/embeddings.npy")
with open("metadata/metadata.json") as f:
    meta = json.load(f)

text = input("Enter search text: ")
with torch.no_grad():
    text_feat = model.encode_text(clip.tokenize([text]).to(device))
    text_feat /= text_feat.norm(dim=-1, keepdim=True)
    text_feat = text_feat.cpu().numpy()[0]

# Compute cosine similarity
sims = embs @ text_feat
top_idx = np.argsort(-sims)[:5]

print("\nTop matches:")
for i in top_idx:
    print(f"{meta[i]['filename']} — score {sims[i]:.3f}")
