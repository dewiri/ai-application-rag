#Automatisiert die Evaluation der Modellantworten
# lädt Reihe von Fragen (ohne Spielvariantentyp)
# generiert für jede Frage eine Antwort
# prüft, ob das erwartete Keyword in der Antwort vorkommt
# Ergebnis wird speichert

import sys
import os
import json
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.retrieval import retrieve
from src.api_client import client

# Lade Fragen & erwartete Keywords
with open("eval/questions.json", "r") as f:
    questions = json.load(f)

# Modell
model = "llama3-70b-8192"

# Ergebnisse zählen
total_keywords = 0
matched_keywords = 0
results = []

for i, entry in enumerate(questions, 1):
    question = entry["question"]
    expected_keywords = [entry["expected_keyword"]]

    docs = retrieve(question, top_k=5)
    context = "\n\n".join(docs)

    prompt = f"""Answer the following question based on the given context.

==================== Context =====================
{context}

==================== Question =====================
{question}

==================== Answer ===================="""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful expert on the rules of Catan."},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content.strip().lower()
    match_count = sum(1 for kw in expected_keywords if kw.lower() in answer)
    matched_keywords += match_count
    total_keywords += len(expected_keywords)

    print(f"[{i}] Q: {question}")
    print(f"    ✔ Keywords matched: {match_count} / {len(expected_keywords)}\n")

    # Ergebnis speichern
    results.append({
        "question": question,
        "expected_keyword": expected_keywords[0],
        "answer": answer,
        "matched": match_count == len(expected_keywords)
    })

# Ergebnisdatei schreiben
output_path = Path("eval/results.json")
output_path.parent.mkdir(exist_ok=True)
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

accuracy = matched_keywords / total_keywords * 100
print(f"Evaluation completed: {matched_keywords} / {total_keywords} keywords matched ({accuracy:.2f}%)")
print(f"Results saved to: {output_path}")