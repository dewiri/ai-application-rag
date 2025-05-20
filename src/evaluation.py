# src/evaluation.py

import os
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import numpy as np
import random
from src.ingestion import load_and_chunk
from src.embeddings import embed_texts
from src.retrieval import build_faiss_index, retrieve, hyde_expand
from src.rerank import rerank

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_questions(chunks: list[str], num_questions: int = 50) -> list[str]:
    """
    Generiert per GPT-4 eine Liste von num_questions aus zufälligen Chunks.
    """
    sample = random.sample(chunks, min(len(chunks), num_questions))
    prompt = (
        "Generate a JSON array of concise, single-sentence questions "
        f"based on the following rule passages:\n\n{sample}"
    )
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    # Annahme: Antwort ist ein JSON-Array von Strings
    return resp.choices[0].message.content  # hier ggf. json.loads

def evaluate(num_questions: int = 50):
    """
    Führt End-to-End-Evaluation durch und gibt Precision/Recall/F1 aus.
    """
    chunks = load_and_chunk("data")
    index = build_faiss_index(embed_texts(chunks))
    questions = generate_questions(chunks, num_questions)

    tp = fp = fn = 0
    for q in questions:
        # Retrieval + HyDE
        pseudo = hyde_expand(q)
        q_emb = embed_texts([q])[0]
        p_emb = embed_texts([pseudo])[0]
        hits = retrieve(index, np.array(q_emb), top_k=20) + \
               retrieve(index, np.array(p_emb), top_k=20)
        docs = [chunks[i] for i,_ in {h[0]:h[1] for h in hits}.items()]
        top_docs = rerank(q, docs, top_n=3)

        # Bot-Antwort via Chat
        prompt = (
            "Answer based solely on the following passages:\n"
            f"{top_docs}\nQuestion: {q}"
        )
        ans = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt}]
        ).choices[0].message.content

        # Autom. Bewertung (relevant/nicht)
        eval_prompt = (
            f"Given the question: \"{q}\" and the answer: \"{ans}\", "
            "was the answer relevant? Reply with YES or NO."
        )
        verdict = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": eval_prompt}]
        ).choices[0].message.content.strip().upper()

        if verdict == "YES":
            tp += 1
        else:
            fp += 1
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / num_questions
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

if __name__ == "__main__":
    evaluate(50)