import sys
import os
import json
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval import retrieve
from src.api_client import client

INPUT_FILE = "eval/questions_with_variants.json"
OUTPUT_FILE = "eval/results_with_variants.json"
MODEL = "llama3-70b-8192"

with open(INPUT_FILE, "r") as f:
    questions = json.load(f)

results = []

for i, entry in enumerate(questions, 1):
    question = entry["question"]
    keyword = entry["expected_keyword"].lower()
    variant = entry["variant"]

    print(f"[{i}] Retrieving for: {question} ({variant})")

    docs = retrieve(question, top_k=5, variant=variant)
    context = "\n\n".join(docs)

    prompt = f"""Answer the following question based on the given context.

==================== Context =====================
{context}

==================== Question =====================
{question}

==================== Answer ===================="""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": f"You are a helpful assistant that uses only the rules from the '{variant}' variant of Catan to answer questions."
            },
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content.strip().lower()
    matched = keyword in answer

    results.append({
        "question": question,
        "variant": variant,
        "expected_keyword": keyword,
        "answer": answer,
        "matched": matched
    })

# Save results
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Evaluation completed. Saved to {OUTPUT_FILE}")