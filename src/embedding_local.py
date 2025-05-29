# dient zur lokalen Berechnung von Embeddings für Texte

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # klein & schnell

def embed_texts(texts: list[str]) -> list[list[float]]:
    return model.encode(texts, show_progress_bar=True)