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

def save_faiss_index(
    vectors: np.ndarray,
    metadata: list[str],
    variant: str,
    dim: int = 384
):
    """
    Speichert den FAISS-Index und die Metadaten für eine Spielvariante.
    
    Parameter:
    - vectors: Die eingebetteten Vektoren (2D-Array)
    - metadata: Die zugehörigen Texte oder Dokumente
    - variant: Spielvariante (z. B. "basegame")
    - dim: Dimension der Embeddings (Standard: 384)
    """
    path = get_variant_path(variant)
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    faiss.write_index(index, str(path / "index.faiss"))

    with open(path / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ FAISS index saved for variant '{variant}'")

def load_faiss_index(variant: str):
    """
    Lädt den FAISS-Index und die Metadaten für eine bestimmte Spielvariante.
    
    Rückgabe:
    - FAISS Index
    - Liste der Metadaten
    """
    path = get_variant_path(variant)
    index = faiss.read_index(str(path / "index.faiss"))
    with open(path / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return index, metadata