# tests/test_index.py

from src.ingestion import load_and_chunk
from embedding_local import embed_texts
from src.retrieval import build_faiss_index, retrieve

def test_faiss_index_and_retrieve():
    dummy_chunks = ["Build a city next to a road.", "No building on desert tiles."]
    embs = embed_texts(dummy_chunks)
    index = build_faiss_index(embs)
    assert index.ntotal == len(embs)

    q_emb = embed_texts(["Can I build on a road?"])[0]
    hits = retrieve(index, q_emb, top_k=5)
    assert isinstance(hits, list)