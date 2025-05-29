# Verarbeitung der PDF's. Erzeugung der Chunks und Embeddings. 
import os
import pickle
import numpy as np
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from embedding_local import embed_texts
from faiss_store import save_faiss_index

# Spielvarianten mit zugehörigen Schlüsselwörtern zur Filterung von PDFs
VARIANT_MAP = {
    "basegame": ["base", "main"],
    "cities_knights": ["cities", "knights", "c_k"],
    "explorers_pirates": ["explorers", "pirates", "e_p"],
    "seafarers": ["seafarers"],
    "traders_barbarians": ["traders", "barbarians", "t_b"]
}

PDF_DIR = Path("data")             # Pfad zum Verzeichnis mit PDF-Dateien
VECTOR_DIR = Path("vectorstore")   # Zielverzeichnis für FAISS-Indizes und Chunks

def load_documents_for_variant(variant_key):
    """
    Lädt alle PDF-Dokumente, die zur Spielvariante passen.
    """
    keywords = VARIANT_MAP[variant_key]
    documents = []
    for pdf_path in PDF_DIR.glob("*.pdf"):
        if any(kw in pdf_path.name.lower() for kw in keywords):
            print(f"Loading: {pdf_path.name}")
            loader = PyPDFLoader(str(pdf_path))
            documents.extend(loader.load())  # Lädt und konvertiert jede Seite in ein Dokument
    return documents

def chunk_documents(documents):
    """
    Zerlegt Dokumente in überlappende Text-Abschnitte (Chunks), möglichst semantisch sinnvoll.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    return splitter.split_documents(documents)

def save_chunks(chunks, path):
    """
    Speichert die erzeugten Text-Chunks als Pickle-Datei.
    """
    with open(path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved {len(chunks)} chunks to {path}")

def main():
    """
    Führt den Ingestion-Prozess für alle Spielvarianten aus:
    - PDF-Dokumente laden
    - in Chunks zerlegen
    - Embeddings berechnen
    - FAISS-Index und Chunks speichern
    """
    for game_variant in VARIANT_MAP.keys():
        print(f"\nProcessing variant: {game_variant}")
        documents = load_documents_for_variant(game_variant)
        print(f"Loaded {len(documents)} documents")

        if not documents:
            print("No matching PDFs found. Skipping this variant.")
            continue

        chunks = chunk_documents(documents)
        print(f"Created {len(chunks)} chunks")

        texts = [chunk.page_content for chunk in chunks]
        vectors = embed_texts(texts)  # Embeddings erzeugen

        variant_dir = VECTOR_DIR / game_variant
        variant_dir.mkdir(exist_ok=True)

        save_chunks(chunks, variant_dir / "chunks.pkl")
        save_faiss_index(np.array(vectors), texts, game_variant)

    print("\nIngestion completed for all variants.")

if __name__ == "__main__":
    main()