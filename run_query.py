from src.retrieval import retrieve

query = input("🔍 Deine Frage: ")
results = retrieve(query)
print("\n📄 Ähnliche Textstellen:\n")
for i, r in enumerate(results, 1):
    print(f"{i}. {r[:300]}\n")