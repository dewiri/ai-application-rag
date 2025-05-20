# tests/test_rerank.py

from src.rerank import rerank

def test_rerank_basic():
    query = "Example query"
    docs = ["Doc related to example query.", "Irrelevant document."]
    top = rerank(query, docs, top_n=1)
    assert isinstance(top, list)
    assert len(top) == 1
    assert "query" in top[0].lower()