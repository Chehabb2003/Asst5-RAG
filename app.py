import streamlit as st
import numpy as np
import pandas as pd
import time

from sentence_transformers import SentenceTransformer
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from transformers import CLIPProcessor, CLIPModel
import torch

# ───────────────────────────────────────────
# 1.  LOAD DATA
# ───────────────────────────────────────────
df = pd.read_csv("transcript_with_index.csv")          # chunk_id, start, end, text, embedding_index

# Text embeddings
text_embs = np.load("text_embeddings.npy").astype("float32")
# Image embeddings
image_embs = np.load("image_embeddings.npy").astype("float32")
# Frame-to-chunk mapping
frame_map = pd.read_csv("frame_to_chunk.csv")      # frame_path, timestamp, chunk_id

# ───────────────────────────────────────────
# 2.  BUILD FAISS INDEX FOR TEXT
# ───────────────────────────────────────────
faiss.normalize_L2(text_embs)
dim = text_embs.shape[1]
text_index = faiss.IndexFlatIP(dim)
text_index.add(text_embs)

# ───────────────────────────────────────────
# 3.  LOAD MODELS & VECTORIZERS (CPU-only)
# ───────────────────────────────────────────
# Text encoder (MPNet)
text_model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2",
    device="cpu"
)
# Lexical baselines
vectorizer   = TfidfVectorizer().fit(df['text'])
tfidf_matrix = vectorizer.transform(df['text'])
bm25         = BM25Okapi([txt.split() for txt in df['text']])
# CLIP for image retrieval (text -> image) using vit-large-patch14
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cpu")
# Normalize image embeddings
image_embs = image_embs / np.linalg.norm(image_embs, axis=1, keepdims=True)

# ───────────────────────────────────────────
# 4.  RETRIEVAL FUNCTIONS
# ───────────────────────────────────────────
def faiss_search(query: str, k: int = 3):
    q_emb = text_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = text_index.search(q_emb, k)
    return [
        dict(chunk_id=int(df.iloc[idx].chunk_id),
             start=float(df.iloc[idx].start),
             end=float(df.iloc[idx].end),
             text=df.iloc[idx].text,
             score=float(D[0][r]))
        for r, idx in enumerate(I[0])
    ]


def tfidf_search(query: str, k: int = 3):
    q_vec  = vectorizer.transform([query])
    scores = (tfidf_matrix @ q_vec.T).toarray().ravel()
    idxs   = np.argsort(scores)[::-1][:k]
    return [
        dict(chunk_id=int(df.iloc[i].chunk_id),
             start=float(df.iloc[i].start),
             end=float(df.iloc[i].end),
             text=df.iloc[i].text,
             score=float(scores[i]))
        for i in idxs
    ]


def bm25_search(query: str, k: int = 3):
    scores = bm25.get_scores(query.split())
    idxs   = np.argsort(scores)[::-1][:k]
    return [
        dict(chunk_id=int(df.iloc[i].chunk_id),
             start=float(df.iloc[i].start),
             end=float(df.iloc[i].end),
             text=df.iloc[i].text,
             score=float(scores[i]))
        for i in idxs
    ]


def clip_image_search(query: str, k: int = 3):
    # Encode query using CLIP text encoder
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_feats = clip_model.get_text_features(**inputs)
    text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
    q_emb = text_feats.cpu().numpy()[0]
    # Compute cosine similarity with image embeddings
    sims = image_embs @ q_emb
    idxs = np.argsort(sims)[::-1][:k]
    results = []
    for idx in idxs:
        row = frame_map.iloc[idx]
        results.append({
            "frame_path": row.frame_path,
            "timestamp": float(row.timestamp),
            "chunk_id": int(row.chunk_id),
            "score": float(sims[idx])
        })
    return results

# ───────────────────────────────────────────
# 5.  STREAMLIT UI
# ───────────────────────────────────────────
st.set_page_config(page_title="Video QA RAG", layout="wide")
st.title("🎥 Multimodal RAG Video QA")

# Embed YouTube video
st.video("https://www.youtube.com/watch?v=dARr3lGKwk8")

method = st.sidebar.selectbox(
    "Retrieval Method", ["FAISS", "TF-IDF", "BM25", "Image"]
)
top_k = st.sidebar.slider("Top-k results", 1, 5, 3)

query = st.text_input("Your Question:")
if st.button("Search") and query:
    tic = time.perf_counter()
    if method == "FAISS":
        hits = faiss_search(query, top_k)
    elif method == "TF-IDF":
        hits = tfidf_search(query, top_k)
    elif method == "BM25":
        hits = bm25_search(query, top_k)
    else:
        hits = clip_image_search(query, top_k)
    latency_ms = (time.perf_counter() - tic) * 1000

    st.write(f"**Results ({method}) · Latency: {latency_ms:.1f} ms**")
    if method == "Image":
        for h in hits:
            st.image(h['frame_path'], caption=f"Frame @ {h['timestamp']:.1f}s — score {h['score']:.3f}")
            chunk = df[df['chunk_id'] == h['chunk_id']].iloc[0]
            st.markdown(f"> {chunk.text}\n")
    else:
        if hits:
            for h in hits:
                st.markdown(
                    f"- **Chunk {h['chunk_id']}** "
                    f"({h['start']:.1f}s – {h['end']:.1f}s) — score {h['score']:.3f}"
                )
                st.markdown(f"> {h['text']}\n")
        else:
            st.warning("No relevant segment found.")
