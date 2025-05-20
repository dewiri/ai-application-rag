# src/ingestion.py

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter

def parse_pdf(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def load_and_chunk(pdf_folder: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = []
    for pdf_path in Path(pdf_folder).glob("*.pdf"):
        content = parse_pdf(str(pdf_path))
        chunks.extend(splitter.split_text(content))
    return chunks

if __name__ == "__main__":
    chunks = load_and_chunk("data")
    print(f"Generated {len(chunks)} chunks.")