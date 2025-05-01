import streamlit as st
import numpy as np
import pandas as pd
import time

from sentence_transformers import SentenceTransformer
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

# ───────────────────────────────────────────
# 1.  LOAD DATA
# ───────────────────────────────────────────
df = pd.read_csv("transcript_with_index.csv")          # chunk_id, start, end, text, embedding_index
text_embs = np.load("text_embeddings.npy").astype("float32")

# ───────────────────────────────────────────
# 2.  BUILD FAISS INDEX (cosine sim via L2-norm)
# ───────────────────────────────────────────
faiss.normalize_L2(text_embs)
dim = text_embs.shape[1]
faiss_index = faiss.IndexFlatIP(dim)
faiss_index.add(text_embs)

# ───────────────────────────────────────────
# 3.  LOAD MODELS & VECTORIZERS (CPU-only)
# ───────────────────────────────────────────
text_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"                       # force CPU so no CUDA/MPS issues
)

vectorizer    = TfidfVectorizer().fit(df["text"])
tfidf_matrix  = vectorizer.transform(df["text"])
bm25          = BM25Okapi([txt.split() for txt in df["text"]])

# ───────────────────────────────────────────
# 4.  RETRIEVAL HELPERS
# ───────────────────────────────────────────
def faiss_search(query: str, k: int = 3):
    q_emb = text_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = faiss_index.search(q_emb, k)
    return [
        dict(chunk_id=int(df.iloc[idx].chunk_id),
             start=float(df.iloc[idx].start),
             end=float(df.iloc[idx].end),
             text=df.iloc[idx].text,
             score=float(D[0][rank]))
        for rank, idx in enumerate(I[0])
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

# ───────────────────────────────────────────
# 5.  STREAMLIT UI
# ───────────────────────────────────────────
st.set_page_config(page_title="Video QA RAG", layout="wide")
st.title("🎥 Multimodal RAG Video QA")

# embed YouTube video
st.video("https://www.youtube.com/watch?v=dARr3lGKwk8")

method = st.sidebar.selectbox(
    "Retrieval Method", ["FAISS", "TF-IDF", "BM25"]
)
top_k = st.sidebar.slider("Top-k results", 1, 5, 3)

query = st.text_input("Your Question:")
if st.button("Search") and query:
    tic = time.perf_counter()
    if method == "FAISS":
        hits = faiss_search(query, top_k)
    elif method == "TF-IDF":
        hits = tfidf_search(query, top_k)
    else:
        hits = bm25_search(query, top_k)
    latency_ms = (time.perf_counter() - tic) * 1000

    if hits:
        st.write(f"**Results ({method}) · Latency: {latency_ms:.1f} ms**")
        for h in hits:
            st.markdown(
                f"- **Chunk {h['chunk_id']}** "
                f"({h['start']:.1f}s – {h['end']:.1f}s) "
                f"— score {h['score']:.3f}"
            )
            st.markdown(f"> {h['text']}\n")
    else:
        st.warning("No relevant segment found.")

