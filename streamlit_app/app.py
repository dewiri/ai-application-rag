# streamlit_app/app.py

import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import numpy as np
from openai import OpenAI

from src.ingestion import load_and_chunk
from src.embeddings import embed_texts
from src.retrieval import build_faiss_index, retrieve, hyde_expand
from src.rerank import rerank

# Init OpenAI-Client
oa = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Catan Rule Expert", layout="wide")
st.sidebar.title("Catan Rule Expert Bot")
model_choice = st.sidebar.selectbox("Choose model", ["gpt-4", "gpt-3.5-turbo"])
query = st.text_input("Ask a Catan rule question:")

@st.cache_resource
def init_index():
    chunks = load_and_chunk("data")
    embs = embed_texts(chunks)
    return build_faiss_index(embs), chunks

index, all_chunks = init_index()

if st.button("Submit") and query:
    # HyDE + Retrieval
    pseudo = hyde_expand(query)
    q_emb = oa.embeddings.create(model="text-embedding-ada-002", input=[query]).data[0].embedding
    p_emb = oa.embeddings.create(model="text-embedding-ada-002", input=[pseudo]).data[0].embedding

    hits_q = retrieve(index, np.array(q_emb), top_k=20)
    hits_p = retrieve(index, np.array(p_emb), top_k=20)
    ids = list({i for i,_ in hits_q+hits_p})
    docs = [all_chunks[i] for i in ids]

    # Reranking
    top3 = rerank(query, docs, top_n=3)

    # Prompt & Antwort
    info = "\n\n".join(top3)
    prompt = (
        "Answer solely based on the following rules:\n"
        f"<information>{info}</information>\n\nQuestion: {query}"
    )
    ans = oa.chat.completions.create(
        model=model_choice,
        messages=[{"role": "system", "content": prompt}]
    ).choices[0].message.content

    st.markdown(f"**Answer:**  {ans}")