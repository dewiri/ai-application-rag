import os
import pickle
import numpy as np
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from embedding_local import embed_texts
from faiss_store import save_faiss_index

# Spielvarianten-Definition – mit erweiterten Schlüsselwörtern
VARIANT_MAP = {
    "basegame": ["base", "main"],
    "cities_knights": ["cities", "knights", "c_k"],
    "explorers_pirates": ["explorers", "pirates", "e_p"],
    "seafarers": ["seafarers"],
    "traders_barbarians": ["traders", "barbarians", "t_b"]
}

PDF_DIR = Path("data")
VECTOR_DIR = Path("vectorstore")

def load_documents_for_variant(variant_key):
    keywords = VARIANT_MAP[variant_key]
    documents = []
    for pdf_path in PDF_DIR.glob("*.pdf"):
        if any(kw in pdf_path.name.lower() for kw in keywords):
            print(f"Loading: {pdf_path.name}")
            loader = PyPDFLoader(str(pdf_path))
            documents.extend(loader.load())
    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    return splitter.split_documents(documents)

def save_chunks(chunks, path):
    with open(path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved {len(chunks)} chunks to {path}")

def main():
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
        vectors = embed_texts(texts)

        variant_dir = VECTOR_DIR / game_variant
        variant_dir.mkdir(exist_ok=True)

        save_chunks(chunks, variant_dir / "chunks.pkl")
        save_faiss_index(np.array(vectors), texts, game_variant)

    print("\nIngestion completed for all variants.")

if __name__ == "__main__":
    main()