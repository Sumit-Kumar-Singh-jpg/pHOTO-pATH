# face_indexer.py
import os, sys, json, argparse
from glob import glob

import numpy as np
import face_recognition
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

PHOTO_DIR = "photos"
META_DIR  = "metadata"
EMB_PATH  = os.path.join(META_DIR, "faces.npy")
META_PATH = os.path.join(META_DIR, "face_meta.json")

VALID_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")

def load_meta():
    if not os.path.exists(META_PATH): return []
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_meta(meta):
    os.makedirs(META_DIR, exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def load_emb():
    if not os.path.exists(EMB_PATH): return None
    return np.load(EMB_PATH)

def save_emb(arr):
    os.makedirs(META_DIR, exist_ok=True)
    np.save(EMB_PATH, arr)

def list_images():
    out = []
    for root, _, files in os.walk(PHOTO_DIR):
        for fn in files:
            if fn.lower().endswith(VALID_EXT):
                out.append(os.path.join(root, fn))
    out.sort()
    return out

def encode_image(path):
    """Return [(box, enc), ...], box as (top,right,bottom,left)."""
    try:
        img = face_recognition.load_image_file(path)
    except (UnidentifiedImageError, OSError, FileNotFoundError):
        return []

    # try CNN first
    boxes = face_recognition.face_locations(img, model="cnn")
    if len(boxes) == 0:
        boxes = face_recognition.face_locations(img, model="hog")

    if len(boxes) == 0:
        return []

    encs = face_recognition.face_encodings(img, boxes)
    return list(zip(boxes, encs))

def main(append_only=True):
    os.makedirs(META_DIR, exist_ok=True)
    meta = load_meta()
    emb  = load_emb()

    known_files = set(m.get("filename") for m in meta)
    imgs = list_images()

    # If not append_only, rebuild from scratch
    if not append_only:
        meta = []
        emb  = None
        known_files = set()

    new_count = 0
    enc_list = [] if emb is None else [emb]

    for path in tqdm(imgs, desc="Indexing"):
        # If file already processed & we are append_only, skip only if *all* faces from this file exist.
        if append_only and path in known_files:
            continue

        pairs = encode_image(path)
        if not pairs:
            continue

        for (top, right, bottom, left), enc in pairs:
            meta.append({
                "filename": path.replace("\\", "/"),
                "box": [int(top), int(left), int(bottom), int(right)],
                "name": meta_name_default(path, top, left, bottom, right)  # empty by default
            })
            enc_list.append(enc.reshape(1, -1))
            new_count += 1

    if new_count == 0:
        print("No new faces found.")
        return 0

    # stack embeddings
    embs = np.vstack(enc_list)
    save_emb(embs)
    save_meta(meta)
    print(f"✅ Indexed {new_count} new faces. Total: {len(meta)}")
    return new_count

def meta_name_default(path, t, l, b, r):
    # keep empty label by default (user labels in UI)
    return ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--append-only", action="store_true", help="Only add new photos (default).")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild metadata and embeddings from scratch.")
    args = parser.parse_args()
    main(append_only=not args.rebuild)
