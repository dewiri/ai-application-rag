# ai-application-rag

## Project Description

This is a **Retrieval-Augmented Generation (RAG)** application that acts as an expert assistant for the board game **Catan**, including the **base game** and all major **expansions**.

Users can ask natural-language questions like:

> _"Can I build a city next to my own road?"_

The system retrieves relevant rule passages and uses a language model to generate **accurate, context-based answers**.

---

## Name & URL

| Name                  | URL / Info                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| Streamlit App         | [catan-rulebot.streamlit.app](https://catan-rulebot.streamlit.app)         |
| Embedding Model Page  | [`all-MiniLM-L6-v2`](https://www.sbert.net/docs/pretrained_models.html)    |
| Code Repository       | [GitHub Repository](https://github.com/dewiri/ai-application-rag)          |
| LLM Backend           | [Groq API (LLaMA 3)](https://console.groq.com/docs)                         |

---

## Data Sources

| Rulebook Version                                | Description          | Link                                                                 |
|--------------------------------------------------|----------------------|----------------------------------------------------------------------|
| CATAN – Base Game (3–4 Players)                  | Main rulebook        | [PDF](https://www.catan.com/sites/default/files/2021-06/catan_base_rules_2020_200707.pdf) |
| CATAN – Base Game (5–6 Players)                  | Extension rules      | [PDF](https://www.catan.com/sites/default/files/2024-03/Catan%20Game%205-6%20Rules%202022%20240313.pdf) |
| CATAN – Seafarers (3–4 Players)                  | Rules & Scenarios    | [PDF](https://www.catan.com/sites/default/files/2021-06/catan-seafarers_2021_rule_book_201201.pdf) |
| CATAN – Seafarers (5–6 Players)                  | Extension rules      | [PDF](https://www.catan.com/sites/default/files/2024-03/Catan%20Seafarers%205-6%202023%20Rules%20220313.pdf) |
| CATAN – Cities & Knights (3–4 Players)           | Rules & Almanac      | [PDF](https://www.catan.com/sites/default/files/2021-06/catan_c_k_2020_rule_book_200708.pdf) |
| CATAN – Cities & Knights (5–6 Players)           | Extension rules      | [PDF](https://www.catan.com/sites/default/files/2024-03/Catan%20C%26K%205-6%202023%20Rules%20240313.pdf) |
| CATAN – Traders & Barbarians (3–4 Players)       | Rules                | [PDF](https://www.catan.com/sites/default/files/2021-06/catan-t_b_2020_rule_book_200820.pdf) |
| CATAN – Traders & Barbarians (5–6 Players)       | Extension rules      | [PDF](https://www.catan.com/sites/default/files/2024-03/Catan%20T%26B%205-6%202020%20Rules%20240313.pdf) |
| CATAN – Explorers & Pirates (3–4 Players)        | Rules                | [PDF](https://www.catan.com/sites/default/files/2021-06/catan_e_p_2020_merged_200707.pdf) |
| CATAN – Explorers & Pirates (5–6 Players)        | Extension rules      | [PDF](https://www.catan.com/sites/default/files/2024-03/Catan%20E%26P%205-6%202022%20Rules%20240313.pdf) |

---

## RAG Improvements

| Aspect                        | Before (Single-Variant Retrieval)                            | Now (Variant-Aware Retrieval + Cross-Variant Analysis)                  |
|-------------------------------|--------------------------------------------------------------|-------------------------------------------------------------------------|
| Vector Store Setup             | One single FAISS index for all game rules combined           | Separate FAISS index per game variant (basegame, seafarers, etc.)       |
| Query Execution               | Search performed on the combined index                        | Search primarily on selected variant’s index                            |
| Cross-Variant Comparison      | Not available                                                | Question also compared across all variant indexes for similarity        |
| Hybrid Retrieval              | Not implemented                                              | Combines semantic similarity and keyword matching within variant       |
| Result Precision             | Lower precision, generic answers                             | Higher precision, variant-specific answers                             |
| Transparency                 | No insight into variant differences                         | Shows similarity scores across all variants to highlight rule differences |
| User Experience              | Simpler but less accurate                                    | More informative and accurate, supports complex queries                 |
| Chunking Strategy         | Naive fixed-length splits, often incoherent                   | Sentence-aware, overlapping chunks via spaCy sentence detection     |

>Together, these upgrades lead to more accurate, explainable, and variant-specific answers — a major step beyond naive, single-index retrieval.
---

## Chunking

### Chunking Strategy

| Version | Method                              | Configuration                         | Description                                                                                                                                                                      |
|---------|--------------------------------------|----------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| V1.0    | Character-based                      | Fixed size only                        | Naive splitting into fixed-length chunks without considering sentence or paragraph boundaries. **Discarded** due to incoherent chunks.                                          |
| V2.0    | SentenceTransformersTokenTextSplitter | Max 256 tokens                         | Token-based semantic splitting. **Not used in the final system** because it created many small chunks unsuitable for rule-heavy texts.                                          |
| V3.0    | RecursiveCharacterTextSplitter       | 1000 characters, 100 overlap           | Paragraph-aware splitting that maintains sentence structure and coherence.                                                                |
| V3.1    | Variant-Aware Chunk Mapping          | Based on filename keywords             | Documents are automatically assigned to game variants (`basegame`, `seafarers`, etc.)                                                     |
| V3.2    | Semantic Filtering + Metadata        | Embedding + variant metadata           | Chunks include metadata for variant, enabling hybrid scoring and cross-variant comparisons.                                               |

> **V3.0–V3.2** are used in combination to enable clean, modular, and context-aware chunking and retrieval.

---

## Choice of LLM

| Name         | Link                                      |
|--------------|-------------------------------------------|
| LLaMA 3 via Groq | [Groq LLM API](https://console.groq.com/docs) |

---

## Hybrid Retrieval Enhancement

To increase the relevance and accuracy of retrieved rulebook passages, a **hybrid retrieval strategy** was implemented. It combines:

- **Dense Semantic Retrieval**: Using vector embeddings and FAISS for similarity
- **Sparse Keyword Matching**: Using Jaccard similarity of token sets (incl. synonym expansion)

### Scoring Formula
- **α = 0.8** by default (80% semantic, 20% keyword)
- Scores are capped at 1.0
- Bonus: slight boost if expected keywords are present in the chunk

### Benefits

- **Improves precision** for rule-related queries
- **Balances meaning and terminology**
- **Highlights scoring components** for transparency in UI
- **Variant-aware** logic: combines with metadata filtering and score boosting

### Real Output Scores (Example Query)

| Chunk | Hybrid Score | Dense Score | Keyword Score |
|-------|--------------|-------------|---------------|
| 1     | 0.389        | 0.617       | 0.047         |
| 2     | 0.384        | 0.615       | 0.037         |
| 3     | 0.376        | 0.600       | 0.039         |

> Perfect score of 1.0 is rare in natural language queries.  
> In practice, **dense scores ≥ 0.60** and **hybrid scores ~0.38** already indicate strong retrieval performance.

---

## Evaluation & Testing

- **Dataset:** 40 real Catan-related questions
- **Approach:** Check if expected keyword appears in the model output
- **Result:**

| Model/Method                                                        | Accuracy | Precision | Recall | F1-Score|
|---------------------------------------------------------------------|----------|-----------|--------|--------|
| Initial Evaluation (Single Vectorstore)                             | 65.0%   | 77.8%      |  72.4% | 75.0%  |
| Game variant-specific vectorstores                                  | 77.5%   | 92.3%      | 77.4%  | 84.1%  |

> **Conclusion:**  The introduction of game variant-specific vectorstores significantly improved answer quality. Keyword match accuracy rose from **65.0% to 77.5%**This demonstrates the value of routing queries through variant-aware context retrieval for more accurate and relevant rule-based responses.
---
### Final Evaluation
Per Game Variant, [10 questions](https://github.com/dewiri/ai-application-rag/blob/main/tests/all_questions.py) were posed in the Streamlit app, and for each variant the system answered correctly.

>You can also explore per-question insights in the improved Streamlit application – view the retrieved context, hybrid retrieval analysis, and similarity scores across all game variants to dig deeper into each classification decision. All detailed results and evaluation metrics are live in the Streamlit app—and they’re updated per question

---

## Environment

- **Local setup**: `.env` file with `python-dotenv`
- **Streamlit Cloud**: secrets managed via `.streamlit/secrets.toml`

---

## References

- [Catan Official Rulebooks](https://www.catan.com/service/game-rules)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Groq API Docs](https://console.groq.com/docs)
- [LangChain – Text Splitters](https://docs.langchain.com/docs/components/text-splitters/)
- [Streamlit](https://docs.streamlit.io/)
- [SentenceTransformers](https://www.sbert.net/)
- [PDFPlumber](https://github.com/jsvine/pdfplumber)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
