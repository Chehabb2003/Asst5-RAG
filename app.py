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

# --- Load transcript data & embeddings ---
df = pd.read_csv("transcript_with_index.csv")  # chunk_id, start, end, text, embedding_index
text_embs = np.load("text_embeddings.npy").astype("float32")
faiss.normalize_L2(text_embs)
dim = text_embs.shape[1]
faiss_index = faiss.IndexFlatIP(dim)
faiss_index.add(text_embs)

# --- Load image embeddings & mapping ---
image_embs = np.load("image_embeddings.npy").astype("float32")
frame_map = pd.read_csv("frame_to_chunk.csv")  # frame_path, timestamp, chunk_id, embedding_index if included

# --- Models & vectorizers ---
text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
vectorizer = TfidfVectorizer().fit(df["text"])
tfidf_matrix = vectorizer.transform(df["text"])
bm25 = BM25Okapi([txt.split() for txt in df["text"]])

# CLIP for image-text retrieval
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# Normalize image embeddings
image_embs = image_embs / np.linalg.norm(image_embs, axis=1, keepdims=True)

# --- Retrieval functions ---
def faiss_search(query, k=3):
    q_emb = text_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = faiss_index.search(q_emb, k)
    return [{"chunk_id": int(df.iloc[idx].chunk_id),
             "start": float(df.iloc[idx].start),
             "end": float(df.iloc[idx].end),
             "text": df.iloc[idx].text,
             "score": float(D[0][i])}
            for i, idx in enumerate(I[0])]


def tfidf_search(query, k=3):
    q_vec = vectorizer.transform([query])
    scores = (tfidf_matrix @ q_vec.T).toarray().ravel()
    idxs = np.argsort(scores)[::-1][:k]
    return [{"chunk_id": int(df.iloc[idx].chunk_id),
             "start": float(df.iloc[idx].start),
             "end": float(df.iloc[idx].end),
             "text": df.iloc[idx].text,
             "score": float(scores[idx])}
            for idx in idxs]


def bm25_search(query, k=3):
    tokens = query.split()
    scores = bm25.get_scores(tokens)
    idxs = np.argsort(scores)[::-1][:k]
    return [{"chunk_id": int(df.iloc[idx].chunk_id),
             "start": float(df.iloc[idx].start),
             "end": float(df.iloc[idx].end),
             "text": df.iloc[idx].text,
             "score": float(scores[idx])}
            for idx in idxs]


def clip_search(query, k=3):
    # Encode text query with CLIP text encoder
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_feat = clip_model.get_text_features(**inputs)
    text_feat = text_feat / text_feat.norm(p=2, dim=-1, keepdim=True)
    text_emb = text_feat.cpu().numpy()[0]
    # Compute cosine similarity with image embeddings
    sims = image_embs @ text_emb
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

# --- Streamlit UI ---
st.set_page_config(page_title="Video QA RAG", layout="wide")
st.title("🎥 Multimodal RAG Video QA")

# Video embed (YouTube instead of local file)
VIDEO_YT_URL = "https://www.youtube.com/watch?v=dARr3lGKwk8"
st.video(VIDEO_YT_URL)

# Sidebar settings
method = st.sidebar.selectbox(
    "Retrieval Method", ["FAISS", "TF-IDF", "BM25", "CLIP-Image"]
)
top_k = st.sidebar.slider("Top-k results", min_value=1, max_value=5, value=3)

# Query input
query = st.text_input("Your Question:")
if st.button("Search") and query:
    start_t = time.perf_counter()
    if method == "FAISS":
        hits = faiss_search(query, k=top_k)
    elif method == "TF-IDF":
        hits = tfidf_search(query, k=top_k)
    elif method == "BM25":
        hits = bm25_search(query, k=top_k)
    else:
        hits = clip_search(query, k=top_k)
    latency = (time.perf_counter() - start_t) * 1000

    if hits:
        st.write(f"**Results (method={method}) | Latency: {latency:.1f} ms**")
        # Display results
        for hit in hits:
            if method == "CLIP-Image":
                st.markdown(
                    f"- **Frame:** {hit['frame_path']} @ {hit['timestamp']:.1f}s — score: {hit['score']:.3f}"
                )
                st.image(hit['frame_path'], caption=f"{hit['timestamp']:.1f}s frame")
                # Link to transcript chunk
                chunk = df[df['chunk_id'] == hit['chunk_id']].iloc[0]
                st.markdown(f"> **Transcript:** {chunk.text}")
            else:
                st.markdown(
                    f"- **Chunk {hit['chunk_id']}** ({hit['start']:.1f}s–{hit['end']:.1f}s) — score: {hit['score']:.3f}"
                )
                st.markdown(f"> {hit['text']}\n")
    else:
        st.write("_No relevant segment or frame found._")
