# src/embeddings.py

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# API-Key muss vorhanden sein UND in Anführungszeichen stehen
if not api_key or not api_key.startswith("sk-"):
    raise ValueError(
        "OPENAI_API_KEY is missing or invalid. "
        "Please set it in .env as OPENAI_API_KEY=\"sk-...\""
    )

from openai import OpenAI
import numpy as np

client = OpenAI(api_key=api_key)

def embed_texts(texts: list[str]) -> list[np.ndarray]:
    """
    Erzeugt Embeddings mit text-embedding-ada-002.
    """
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=texts
    )
    return [np.array(item.embedding) for item in response.data]