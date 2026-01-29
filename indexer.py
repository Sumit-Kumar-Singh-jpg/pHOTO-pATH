import os, json
from PIL import Image, UnidentifiedImageError
import numpy as np
import torch
import clip
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1, MTCNN
import cv2

PHOTO_DIR = "photos"
META_DIR = "metadata"
EMBED_FILE = os.path.join(META_DIR, "embeddings.npy")
FACE_FILE = os.path.join(META_DIR, "face_embeddings.npy")
META_FILE = os.path.join(META_DIR, "metadata.json")

os.makedirs(META_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

# Load CLIP
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Load FaceNet models
mtcnn = MTCNN(image_size=160, margin=0, device=device)
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

paths = []
for root, _, files in os.walk(PHOTO_DIR):
    for fn in files:
        if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            paths.append(os.path.join(root, fn))
paths.sort()

clip_embs = []
face_embs = []
meta = []

for p in tqdm(paths, desc="Indexing photos"):
    try:
        img = Image.open(p).convert("RGB")
    except:
        print("Skipping unreadable:", p)
        continue

    # --- CLIP embedding ---
    img_t = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        clip_vec = clip_model.encode_image(img_t)
        clip_vec = clip_vec / clip_vec.norm(dim=-1, keepdim=True)
        clip_vec = clip_vec.cpu().numpy()[0]

    # --- FACE embedding ---
    face_vec = np.zeros(512)  # default no-face vector
    try:
        face = mtcnn(img)
        if face is not None:
            face = face.unsqueeze(0).to(device)
            with torch.no_grad():
                vec = facenet(face)
            face_vec = vec.squeeze().cpu().numpy()
    except Exception:
        pass

    clip_embs.append(clip_vec)
    face_embs.append(face_vec)
    meta.append({"path": p})

np.save(EMBED_FILE, np.array(clip_embs))
np.save(FACE_FILE, np.array(face_embs))

with open(META_FILE, "w") as f:
    json.dump(meta, f, indent=2)

print("✅ Indexing complete")
