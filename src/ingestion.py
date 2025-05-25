import os
import pickle
import numpy as np
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from embedding_local import embed_texts
from faiss_store import save_faiss_index

VECTOR_DIR = Path("vectorstore")
VECTOR_DIR.mkdir(exist_ok=True)

PDF_DIR = Path("data")

def load_documents_from_pdfs(pdf_dir: Path):
    documents = []
    for pdf_path in pdf_dir.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        documents.extend(docs)
    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    return splitter.split_documents(documents)

def save_chunks(chunks, output_file: Path):
    with open(output_file, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved {len(chunks)} chunks to {output_file}")

def main():
    print("Loading PDFs...")
    documents = load_documents_from_pdfs(PDF_DIR)
    print(f"Loaded {len(documents)} documents.")

    print("Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    chunks_path = VECTOR_DIR / "chunks.pkl"
    save_chunks(chunks, chunks_path)

    print("Creating embeddings...")
    texts = [chunk.page_content for chunk in chunks]
    vectors = embed_texts(texts)

    print("Saving FAISS index...")
    save_faiss_index(np.array(vectors), texts)

    print("Done.")

if __name__ == "__main__":
    main()