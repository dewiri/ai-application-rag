from pathlib import Path
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter

def parse_pdf(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def load_and_chunk(folder: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    all_chunks = []
    for file in Path(folder).glob("*.pdf"):
        raw = parse_pdf(str(file))
        chunks = splitter.split_text(raw)
        all_chunks.extend(chunks)
    return all_chunks