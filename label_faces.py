# label_faces.py
import os, sys, json, subprocess, time
from collections import Counter

import numpy as np
from PIL import Image, UnidentifiedImageError
import streamlit as st

# ------------------ PATHS ------------------
PHOTO_DIR = "photos"
META_DIR = "metadata"
FACE_EMBED_FILE = os.path.join(META_DIR, "faces.npy")
FACE_META_FILE  = os.path.join(META_DIR, "face_meta.json")

# ------------------ PAGE + THEME ------------------
st.set_page_config(page_title="Face Manager", layout="wide")

st.markdown("""
<style>
/* Hybrid Modern Blue (mac-like) + Neon Dark (cyber) */
:root {
  --blue1: #0a84ff;      /* mac blue */
  --blue2: #4aa3ff;
  --neon:  #00f0ff;      /* cyber teal */
  --neon2: #ff2bd6;      /* neon magenta */
  --bgd:   #0b0f14;      /* dark base */
  --card:  #111827;      /* dark card */
  --ink:   #e5e7eb;
}
html, body, .stApp {
  background: linear-gradient(180deg, #0b0f14 0%, #0d1220 60%, #0b0f14 100%);
  color: var(--ink);
}
.big-title {
  font-size: 30px; font-weight: 800; letter-spacing: .2px;
  padding: 8px 0 12px 0;
  background: linear-gradient(90deg, var(--blue1), var(--neon2));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.card {
  background: linear-gradient(180deg, rgba(17,24,39,.7), rgba(17,24,39,.95));
  border: 1px solid rgba(255,255,255,.06);
  border-radius: 14px;
  padding: 14px;
  box-shadow: 0 6px 24px rgba(0,0,0,.25);
}
.face-thumb img, .stImage img {
  border-radius: 10px !important;
  box-shadow: 0 10px 24px rgba(0,0,0,.35);
}
.face-thumb:hover { filter: drop-shadow(0 0 10px var(--neon)); opacity: .95; }
.stButton>button {
  border-radius: 10px; font-weight: 600; border: 1px solid rgba(255,255,255,.14);
  transition: all .18s ease-in-out;
  background: linear-gradient(180deg, #0a84ff, #005ae0);
  color: white;
}
.stButton>button:hover { transform: translateY(-1px); box-shadow: 0 8px 26px rgba(10,132,255,.35); }
.primary-ghost .stButton>button {
  background: linear-gradient(180deg, rgba(10,132,255,.1), rgba(10,132,255,.04));
  color: #dbeafe; border: 1px solid rgba(59,130,246,.35);
}
.neon-ghost .stButton>button {
  background: linear-gradient(180deg, rgba(0,240,255,.08), rgba(0,240,255,.02));
  color: #bffcff; border: 1px solid rgba(0,240,255,.35);
}
.warn-ghost .stButton>button {
  background: linear-gradient(180deg, rgba(255,43,214,.08), rgba(255,43,214,.02));
  color: #ffd6f6; border: 1px solid rgba(255,43,214,.35);
}
.metric { text-align:center; padding: 10px; border-radius: 12px; border:1px solid rgba(255,255,255,.06); }
hr { border: none; border-top: 1px solid rgba(128,128,128,0.25); margin: 0.8rem 0; }
</style>
""", unsafe_allow_html=True)

# ------------------ SESSION STATE ------------------
ss = st.session_state
if "view" not in ss: ss.view = "grid"       # grid | person | cluster | export
if "selected_face" not in ss: ss.selected_face = None
if "rename_mode" not in ss: ss.rename_mode = False
if "_needs_rerun" not in ss: ss._needs_rerun = False
if "_toast" not in ss: ss._toast = None

def request_rerun():
    ss._needs_rerun = True

def perform_rerun_if_needed():
    if ss._needs_rerun:
        ss._needs_rerun = False
        st.rerun()

def go(view: str):
    ss.view = view
    if view != "person":
        ss.selected_face = None
        ss.rename_mode = False
    request_rerun()

def focus_face(i: int):
    ss.view = "person"
    ss.selected_face = int(i)
    ss.rename_mode = False
    request_rerun()

# ------------------ IO HELPERS ------------------
def safe_load_meta():
    if not os.path.exists(FACE_META_FILE): return None
    try:
        with open(FACE_META_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load metadata: {e}")
        return None

def safe_save_meta(data):
    try:
        with open(FACE_META_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Failed to write metadata: {e}")
        return False

def open_image(path):
    try:
        return Image.open(path).convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError, OSError):
        return None

def crop_face(img, box):
    if img is None or not box: return None
    try:
        t, l, b, r = map(int, box)
        return img.crop((l, t, r, b))
    except Exception:
        w, h = (img.size if img else (256, 256))
        size = min(w, h); cx, cy = w//2, h//2
        return img.crop((cx-size//2, cy-size//2, cx+size//2, cy+size//2))

# ------------------ LOAD DATA ------------------
face_meta = safe_load_meta()
if face_meta is None:
    st.error("❌ No face metadata found. Click **Ingest new photos** to build it.")
    face_meta = []

faces_emb = None
if os.path.exists(FACE_EMBED_FILE):
    try: faces_emb = np.load(FACE_EMBED_FILE)
    except Exception: faces_emb = None

total = len(face_meta)
labeled = sum(1 for m in face_meta if m.get("name"))
unlabeled = total - labeled

# ------------------ SIDEBAR ------------------
st.sidebar.title("📁 Face Manager")
st.sidebar.caption("Offline • Private • Local")

with st.sidebar:
    st.write(f"✅ **{labeled}** labeled / 🕵 **{total}** faces")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🖼 Gallery", use_container_width=True):
            go("grid")
    with c2:
        if st.button("👥 Clusters", use_container_width=True):
            go("cluster")

    if st.button("📁 Export", use_container_width=True):
        go("export")

    st.markdown("---")
    # Ingest new photos (index + cluster)
    if st.button("➕ Ingest new photos", use_container_width=True):
        # Run face_indexer then cluster_faces via subprocess
        st.toast("Scanning new photos…")
        python_exe = sys.executable
        try:
            with st.status("Indexing faces (CNN → HOG fallback)…", expanded=True) as s:
                r = subprocess.run([python_exe, "face_indexer.py", "--append-only"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                st.write(r.stdout)
                if r.returncode != 0:
                    st.error("Indexing failed. Check console output.")
                    st.stop()
                s.update(label="Clustering faces (DBSCAN)…", state="running")
                r2 = subprocess.run([python_exe, "cluster_faces.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                st.write(r2.stdout)
                if r2.returncode != 0:
                    st.warning("Indexing done, but clustering failed. Labels will still work.")
                s.update(label="Done!", state="complete")
            st.success("Ingest complete. Metadata refreshed.")
            request_rerun()
        except FileNotFoundError:
            st.error("Could not run indexer. Ensure this file sits next to face_indexer.py and cluster_faces.py")
    if st.button("🔄 Reload metadata", use_container_width=True):
        request_rerun()

perform_rerun_if_needed()

# ------------------ HEADER STATS ------------------
st.markdown("<div class='big-title'>Face Manager</div>", unsafe_allow_html=True)
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.markdown(f"<div class='metric'>Total<br/><h3>{total}</h3></div>", unsafe_allow_html=True)
mc2.markdown(f"<div class='metric'>Labeled<br/><h3>{labeled}</h3></div>", unsafe_allow_html=True)
mc3.markdown(f"<div class='metric'>Unlabeled<br/><h3>{unlabeled}</h3></div>", unsafe_allow_html=True)
mc4.markdown(f"<div class='metric'>Clusters<br/><h3>{len({m.get('cluster',-1) for m in face_meta}) if face_meta else 0}</h3></div>", unsafe_allow_html=True)
st.markdown("<hr/>", unsafe_allow_html=True)

# ------------------ GRID ------------------
if ss.view == "grid":
    with st.container():
        st.markdown("#### 🖼 Gallery")
        search = st.text_input("Search by name (empty shows all)", value="")
        items = [(i, m) for i, m in enumerate(face_meta) if (search or "").lower() in m.get("name", "").lower()] \
                if search else list(enumerate(face_meta))
        if not items:
            st.info("No faces match your search.")
        else:
            cols = st.columns(6, gap="small")
            for i, m in items:
                img = open_image(m.get("filename", ""))
                face = crop_face(img, m.get("box"))
                if face is None:
                    face = Image.new("RGB", (160, 160), (210, 210, 210))
                with cols[i % 6]:
                    st.image(face, caption=m.get("name", "(unlabeled)") or "(unlabeled)", width=120)
                    if st.button("Open", key=f"open_{i}"):
                        focus_face(i)
    st.stop()

# ------------------ PERSON DETAIL ------------------
if ss.view == "person":
    idx = ss.selected_face or 0
    if not (0 <= idx < len(face_meta)):
        go("grid")
        perform_rerun_if_needed()
        st.stop()

    meta = face_meta[idx]
    current_name = meta.get("name", "") or ""
    st.markdown(f"### 👤 {current_name or 'Unlabeled'}")

    left, right = st.columns([1.2, 2])
    with left:
        img = open_image(meta.get("filename", ""))
        crop = crop_face(img, meta.get("box"))
        if crop is None:
            crop = Image.new("RGB", (240, 240), (220, 220, 220))
        st.image(crop, width=260)

        existing = sorted({m.get("name", "") for m in face_meta if m.get("name")})
        options = existing + ["➕ Add new person"]
        sel_idx = options.index(current_name) if current_name in options else len(existing)
        choice = st.selectbox("Assign person", options, index=sel_idx, key=f"sel_{idx}")

        if choice == "➕ Add new person":
            new_typed = st.text_input("New name", key=f"new_{idx}")
            final_name = (new_typed or "").strip()
        else:
            final_name = choice.strip()

        s1, s2 = st.columns(2)
        if s1.button("✅ Save for this face", key=f"save_one_{idx}"):
            if final_name:
                face_meta[idx]["name"] = final_name
                if safe_save_meta(face_meta):
                    st.success(f"Saved: {final_name}")
                    request_rerun()
            else:
                st.warning("Enter a name before saving.")

        if s2.button("👥 Apply to all in this group", key=f"apply_group_{idx}"):
            if not final_name:
                st.warning("Choose or type a name first.")
            else:
                if current_name:
                    group_idxs = [i for i, m in enumerate(face_meta) if m.get("name") == current_name]
                else:
                    group_idxs = [i for i, m in enumerate(face_meta) if not m.get("name")]
                for j in group_idxs: face_meta[j]["name"] = final_name
                if safe_save_meta(face_meta):
                    st.success(f"Applied '{final_name}' to {len(group_idxs)} faces.")
                    request_rerun()

        st.markdown("---")
        if current_name:
            if st.button("✏️ Rename this person everywhere", key=f"rename_every_{idx}"):
                ss.rename_mode = True
                request_rerun()
        if ss.rename_mode and current_name:
            new_all = st.text_input("Rename all occurrences to:", value=current_name, key=f"rename_input_{idx}")
            cA, cB = st.columns(2)
            if cA.button("💾 Confirm", key=f"confirm_{idx}"):
                for m in face_meta:
                    if m.get("name") == current_name:
                        m["name"] = new_all.strip()
                if safe_save_meta(face_meta):
                    st.success("Renamed successfully.")
                    ss.rename_mode = False
                    request_rerun()
            if cB.button("Cancel", key=f"cancel_{idx}"):
                ss.rename_mode = False
                request_rerun()

        st.markdown("---")
        if st.button("⬅ Back to gallery", key=f"back_{idx}"):
            go("grid")

    with right:
        st.markdown("#### Related faces")
        if current_name:
            rel = [(i, m) for i, m in enumerate(face_meta) if m.get("name") == current_name]
        else:
            rel = [(i, m) for i, m in enumerate(face_meta) if not m.get("name")]
        if not rel:
            st.info("No related faces found.")
        else:
            cols2 = st.columns(6, gap="small")
            for j, (i_rel, m_rel) in enumerate(rel):
                img2 = open_image(m_rel.get("filename", ""))
                crop2 = crop_face(img2, m_rel.get("box"))
                if crop2 is None:
                    crop2 = Image.new("RGB", (140, 140), (210, 210, 210))
                with cols2[j % 6]:
                    st.image(crop2, width=120)
                    if st.button("Focus", key=f"focus_{i_rel}"):
                        focus_face(i_rel)
                    if m_rel.get("name"):
                        if st.button("Remove label", key=f"rm_{i_rel}"):
                            face_meta[i_rel].pop("name", None)
                            safe_save_meta(face_meta)
                            request_rerun()
                    else:
                        if st.button("Label this", key=f"lb_{i_rel}"):
                            focus_face(i_rel)

    perform_rerun_if_needed()
    st.stop()

# ------------------ CLUSTERS ------------------
if ss.view == "cluster":
    st.markdown("### 👥 Clusters")
    if not face_meta or "cluster" not in (face_meta[0] if face_meta else {}):
        st.warning("No clusters found. Click **Ingest new photos** to build embeddings, then cluster.")
        st.stop()

    clusters = {}
    for i, m in enumerate(face_meta):
        cid = m.get("cluster", -1)
        clusters.setdefault(cid, []).append(i)

    # Unclustered
    noise = clusters.get(-1, [])
    if noise:
        st.subheader(f"🚫 Unclustered — {len(noise)}")
        colsN = st.columns(8, gap="small")
        for j, i in enumerate(noise[:32]):
            m = face_meta[i]; img = open_image(m["filename"]); f = crop_face(img, m["box"])
            f = f or Image.new("RGB", (120,120), (210,210,210))
            with colsN[j % 8]:
                st.image(f, width=90, caption=m.get("name","?") or "?")
                if st.button("Open", key=f"open_noise_{i}"):
                    focus_face(i)
        st.markdown("<hr/>", unsafe_allow_html=True)

    # Clusters by size
    ids = [c for c in clusters if c != -1]
    ids.sort(key=lambda c: -len(clusters[c]))
    for cid in ids:
        members = clusters[cid]
        st.subheader(f"Cluster #{cid} — {len(members)}")
        colsC = st.columns(8, gap="small")
        for k, i in enumerate(members[:16]):
            m = face_meta[i]; img = open_image(m["filename"]); f = crop_face(img, m["box"])
            f = f or Image.new("RGB", (120,120), (210,210,210))
            with colsC[k % 8]:
                st.image(f, width=90, caption=m.get("name","?") or "?")
                if st.button("Open", key=f"open_{cid}_{i}"):
                    focus_face(i)

        names = [face_meta[i].get("name","") for i in members if face_meta[i].get("name")]
        suggested = Counter(names).most_common(1)[0][0] if names else ""
        new_name = st.text_input(f"Name for Cluster #{cid}", value=suggested, key=f"clname_{cid}")
        a1, a2 = st.columns([1,1])
        if a1.button(f"✅ Apply to cluster #{cid}", key=f"apply_{cid}"):
            final = new_name.strip()
            if final:
                for i in members: face_meta[i]["name"] = final
                if safe_save_meta(face_meta):
                    st.success(f"Cluster #{cid} labeled as **{final}**")
                    request_rerun()
            else:
                st.warning("Enter a name before applying.")
        if a2.button("🔄 Refresh", key=f"refresh_{cid}"):
            request_rerun()

    perform_rerun_if_needed()
    st.stop()

# ------------------ EXPORT ------------------
if ss.view == "export":
    st.markdown("### 📁 Export Faces by Person")
    out_dir = st.text_input("Export folder", "exported_faces")
    if st.button("📤 Export"):
        os.makedirs(out_dir, exist_ok=True)
        count = 0
        for i, m in enumerate(face_meta):
            img = open_image(m.get("filename", "")); crop = crop_face(img, m.get("box"))
            if crop is None: continue
            name = (m.get("name","unknown") or "unknown").strip()
            person_dir = os.path.join(out_dir, name)
            os.makedirs(person_dir, exist_ok=True)
            crop.save(os.path.join(person_dir, f"face_{i}.jpg"))
            count += 1
        st.success(f"✅ Exported {count} faces to '{out_dir}/'")

    if st.button("⬅ Back to gallery"):
        go("grid")

    perform_rerun_if_needed()
    st.stop()
