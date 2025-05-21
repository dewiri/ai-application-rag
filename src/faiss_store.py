import faiss
import numpy as np
import pickle
from pathlib import Path

VECTOR_DIR = Path("vectorstore")
VECTOR_DIR.mkdir(exist_ok=True)

def save_faiss_index(vectors: np.ndarray, metadata: list[str], dim: int = 384):
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    faiss.write_index(index, str(VECTOR_DIR / "index.faiss"))

    with open(VECTOR_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

def load_faiss_index():
    index = faiss.read_index(str(VECTOR_DIR / "index.faiss"))
    with open(VECTOR_DIR / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata