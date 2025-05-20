# tests/test_ingestion.py

from src.ingestion import load_and_chunk

def test_ingestion_returns_list_of_strings():
    chunks = load_and_chunk("data")
    assert isinstance(chunks, list)
    assert all(isinstance(c, str) for c in chunks)