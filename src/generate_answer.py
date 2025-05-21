from src.retrieval import retrieve
from src.api_client import client

def build_prompt(query, chunks):
    context = "\n\n".join(chunks)
    return f"""Beantworte die folgende Frage basierend auf dem Kontext unten.

==================== Kontext ====================
{context}

==================== Frage ====================
{query}

==================== Antwort ===================="""

def answer_query(query: str, model="llama3-70b-8192"):
    docs = retrieve(query, top_k=5)
    prompt = build_prompt(query, docs)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Du bist ein hilfreicher Catan-Regel-Experte."},
            {"role": "user", "content": prompt}
        ]
    )

    print("Antwort:\n")
    print(response.choices[0].message.content.strip())

if __name__ == "__main__":
    question = input("Deine Frage: ")
    answer_query(question)