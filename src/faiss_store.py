import faiss
import numpy as np
import pickle
from pathlib import Path

# Hauptverzeichnis für alle Vektorstores
BASE_DIR = Path("vectorstore")
BASE_DIR.mkdir(exist_ok=True)

def get_variant_path(variant: str) -> Path:
    """
    Liefert den Speicherpfad für eine bestimmte Spielvariante.
    Erstellt das Verzeichnis bei Bedarf.
    """
    variant_path = BASE_DIR / variant
    variant_path.mkdir(parents=True, exist_ok=True)
    return variant_path

def normalize(vectors: np.ndarray) -> np.ndarray:
    """
    Normalisiert die Vektoren auf Länge 1 für Cosine Similarity.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-10)  # verhindert Division durch 0

def save_faiss_index(
    vectors: np.ndarray,
    metadata: list[str],
    variant: str,
    dim: int = 384
):
    """
    Speichert einen FAISS Index mit Cosine Similarity.
    """
    path = get_variant_path(variant)
    
    # 🔁 Cosine Similarity = normalisierte L2-Distanz mit innerem Produkt
    vectors = normalize(vectors).astype("float32")
    index = faiss.IndexFlatIP(dim)  # Inner Product für Cosine Similarity
    index.add(vectors)

    faiss.write_index(index, str(path / "index.faiss"))

    with open(path / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ Cosine FAISS index saved for variant '{variant}'")

def load_faiss_index(variant: str):
    """
    Lädt den FAISS Index mit Cosine Similarity.
    """
    path = get_variant_path(variant)
    index = faiss.read_index(str(path / "index.faiss"))
    with open(path / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata