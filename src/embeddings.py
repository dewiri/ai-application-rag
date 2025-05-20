# src/embeddings.py

import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import numpy as np

# Deutscher Kommentar: Initialisiere OpenAI-Client mit API-Key aus .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_texts(texts: list[str]) -> list[np.ndarray]:
    """
    Erzeugt Embeddings mit text-embedding-ada-002.
    """
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=texts
    )
    return [np.array(item.embedding) for item in response.data]