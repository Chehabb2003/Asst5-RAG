import streamlit as st
import numpy as np
import pandas as pd
import time

from sentence_transformers import SentenceTransformer
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

# ───────────────────────────────────────────
# 1. LOAD DATA
# ───────────────────────────────────────────
# CSV must have columns: chunk_id, start, end, text
# transcript_with_index.csv holds the embedding_index mapping

df = pd.read_csv("transcript_with_index.csv")
text_embs = np.load("text_embeddings.npy").astype("float32")

# ───────────────────────────────────────────
# 2. BUILD FAISS INDEX
# ───────────────────────────────────────────
faiss.normalize_L2(text_embs)
dim = text_embs.shape[1]
faiss_index = faiss.IndexFlatIP(dim)
faiss_index.add(text_embs)

# ───────────────────────────────────────────
# 3. LOAD MODELS & VECTORIZERS
# ───────────────────────────────────────────
# Use the state-of-the-art e5-large-v2 encoder
text_model = SentenceTransformer(
    "intfloat/e5-large-v2",
    device="cpu"
)
# Lexical baselines
vectorizer   = TfidfVectorizer().fit(df['text'])
tfidf_matrix = vectorizer.transform(df['text'])
bm25         = BM25Okapi([txt.split() for txt in df['text']])

# ───────────────────────────────────────────
# 4. RETRIEVAL FUNCTIONS (with score filter)
# ───────────────────────────────────────────
def faiss_search(query: str, k: int = 3, min_score: float = 0.35):
    q_emb = text_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = faiss_index.search(q_emb, k)

    results = []
    for rank, (idx, score) in enumerate(zip(I[0], D[0])):
        if score < min_score:           # cosine similarity too small → ignore
            continue
        row = df.iloc[idx]
        results.append({
            "chunk_id": int(row.chunk_id),
            "start":   float(row.start),
            "end":     float(row.end),
            "text":    row.text,
            "score":   float(score),
        })
    return results


def tfidf_search(query: str, k: int = 3, min_score: float = 0.05):
    q_vec  = vectorizer.transform([query])
    scores = (tfidf_matrix @ q_vec.T).toarray().ravel()

    # Filter: keep only docs whose score ≥ max_score * 0.10  *and* ≥ min_score
    max_score = scores.max(initial=0)
    thresh    = max(min_score, 0.10 * max_score)
    idxs      = np.where(scores >= thresh)[0]

    # If some survive, take best-k; else return empty
    idxs = idxs[np.argsort(scores[idxs])[::-1][:k]]
    return [
        {
            "chunk_id": int(df.iloc[i].chunk_id),
            "start":   float(df.iloc[i].start),
            "end":     float(df.iloc[i].end),
            "text":    df.iloc[i].text,
            "score":   float(scores[i]),
        }
        for i in idxs
    ]


def bm25_search(query: str, k: int = 3, min_score: float = 3.0):
    scores = bm25.get_scores(query.split())
    max_score = scores.max(initial=0)
    if max_score < min_score:
        return []                      # nothing even close → reject entirely

    idxs = np.argsort(scores)[::-1][:k]
    return [
        {
            "chunk_id": int(df.iloc[i].chunk_id),
            "start":   float(df.iloc[i].start),
            "end":     float(df.iloc[i].end),
            "text":    df.iloc[i].text,
            "score":   float(scores[i]),
        }
        for i in idxs if scores[i] >= min_score
    ]

# ───────────────────────────────────────────
# 5. STREAMLIT UI
# ───────────────────────────────────────────
st.set_page_config(page_title="Video QA RAG", layout="wide")
st.title("🎥 Multimodal RAG Video QA")

# Embed the YouTube video
st.video("https://www.youtube.com/watch?v=dARr3lGKwk8")

# Sidebar controls
method = st.sidebar.selectbox("Retrieval Method", ["FAISS", "TF-IDF", "BM25"])
top_k  = st.sidebar.slider("Top-k results", 1, 5, 3)

# **NEW**: minimum-score slider, value range depends on method
default_thresh = {"FAISS": 0.35, "TF-IDF": 0.05, "BM25": 3.0}
min_score = st.sidebar.number_input(
    "Min relevance score", min_value=0.0,
    value=default_thresh[method], step=0.01
)


# Question input
query = st.text_input("Your Question:")
if st.button("Search") and query:
    start_t = time.perf_counter()
    if method == "FAISS":
        hits = faiss_search(query, top_k, min_score)
    elif method == "TF-IDF":
        hits = tfidf_search(query, top_k, min_score)
    else:
        hits = bm25_search(query, top_k, min_score)
    latency = (time.perf_counter() - start_t) * 1000

    st.write(f"**Results ({method}) · Latency: {latency:.1f} ms**")
    if hits:
        for h in hits:
            st.markdown(
                f"- **Chunk {h['chunk_id']}** "
                f"({h['start']:.1f}s–{h['end']:.1f}s) — score {h['score']:.3f}"
            )
            st.markdown(f"> {h['text']}\n")
    else:
        st.warning("No relevant segment found.")
