from src.ingestion import load_and_chunk
from src.embedding_local import embed_texts
from src.faiss_store import save_faiss_index
import numpy as np

chunks = load_and_chunk("data")
vectors = embed_texts(chunks)
vectors_np = np.array(vectors).astype("float32")
save_faiss_index(vectors_np, chunks)
print("FAISS-Index erstellt & gespeichert.")