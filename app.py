import streamlit as st
import torch
import clip
from PIL import Image
import numpy as np
import json
import os
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Load model ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ---------- Load metadata ----------
EMB_PATH = "metadata/embeddings.npy"
META_PATH = "metadata/metadata.json"

embs = np.load(EMB_PATH)

with open(META_PATH) as f:
    meta = json.load(f)

# ---------- Utility functions ----------
def get_text_embedding(text):
    """Return CLIP embedding for input text."""
    with torch.no_grad():
        text_features = model.encode_text(clip.tokenize([text]).to(device))
    return text_features.cpu().numpy()


def get_image_embedding(image_path):
    """Return CLIP embedding for an image from a file path."""
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features.cpu().numpy()


def search_by_text(query, top_k=5):
    """Search images by text description."""
    query_emb = get_text_embedding(query)
    sims = cosine_similarity(query_emb, embs)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [(sims[i], meta[i]) for i in top_idx]


def search_by_image(image_path, top_k=5):
    """Search images visually similar to the uploaded image."""
    query_emb = get_image_embedding(image_path)
    sims = cosine_similarity(query_emb, embs)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [(sims[i], meta[i]) for i in top_idx]


# ---------- Streamlit UI ----------
st.title("📸 Offline Photo Finder")

mode = st.radio("Search mode:", ["Text query", "Image upload"])

if mode == "Text query":
    query = st.text_input("Enter search text:")
    if st.button("Search") and query:
        results = search_by_text(query)
        st.subheader(f"Results for '{query}'")
        for score, data in results:
            st.image(
                data["filename"],
                caption=f"{data['filename']} — Score: {score:.4f}",
                use_container_width=True
            )

else:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_path = f"temp_upload.{uploaded_file.name.split('.')[-1]}"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image(image_path, caption="Query Image", use_container_width=True)

        if st.button("Search"):
            results = search_by_image(image_path)
            st.subheader("Visually Similar Photos:")
            for score, data in results:
                st.image(
                    data["filename"],
                    caption=f"{data['filename']} — Score: {score:.4f}",
                    use_container_width=True
                )
        os.remove(image_path)
