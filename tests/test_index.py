# tests/test_index.py

from src.ingestion import load_and_chunk
from src.embeddings import embed_texts
from src.retrieval import build_faiss_index, retrieve

def test_faiss_index_and_retrieve():
    # Simulierter kurzer Datensatz
    dummy_chunks = ["Test text one.", "Another test text."]
    embs = embed_texts(dummy_chunks)
    index = build_faiss_index(embs)
    assert index.ntotal == len(embs)

    # Test-Retrieval
    q_emb = embed_texts(["Test text one."])[0]
    hits = retrieve(index, q_emb, top_k=2)
    assert isinstance(hits, list)