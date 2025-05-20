# src/retrieval.py

import os
from dotenv import load_dotenv
load_dotenv()

import faiss
import numpy as np
from openai import OpenAI

# Für HyDE Query Expansion
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_faiss_index(embeddings: list[np.ndarray], dim: int = 1536) -> faiss.Index:
    """
    Baut einen IVF+PQ FAISS-Index.
    """
    quantizer = faiss.IndexFlatL2(dim)
    nlist = 100
    m = 64
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)
    arr = np.vstack(embeddings).astype('float32')
    index.train(arr)
    index.add(arr)
    return index

def retrieve(
    index: faiss.Index,
    query_emb: np.ndarray,
    top_k: int = 10,
    threshold: float = 0.35
) -> list[tuple[int, float]]:
    """
    Holt top_k Ergebnisse und filtert per Distanz-Threshold.
    Rückgabe: Liste von (Dokument-ID, Distanz).
    """
    dists, ids = index.search(query_emb.reshape(1, -1).astype('float32'), top_k)
    return [(int(idx), float(dist))
            for idx, dist in zip(ids[0], dists[0]) if dist < threshold]

def hyde_expand(query: str) -> str:
    """
    HyDE: Generiere hypothetisches Antwort-Dokument via GPT-3.5.
    """
    prompt = (
        "You are a helpful assistant. "
        f"Generate a concise document passage that could answer: \"{query}\""
    )
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content