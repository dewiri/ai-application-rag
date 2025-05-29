# Catan Rule Expert Bot (RAG)

---

### Project Description

This is a **Retrieval-Augmented Generation (RAG)** application that acts as an expert assistant for the board game **Catan**, including the **base game** and all major **expansions**.

Users can ask natural-language questions like:

> _"Can I build a city next to my own road?"_

The system retrieves relevant rule passages and uses a language model to generate **accurate, context-based answers**.

---

### Repository & Demo

| Name                      | URL / Info                                                               |
|---------------------------|---------------------------------------------------------------------------|
| Streamlit App          | [catan-rulebot.streamlit.app](https://catan-rulebot.streamlit.app)       |
| Embedding Model        | [`all-MiniLM-L6-v2`](https://www.sbert.net/docs/pretrained_models.html) via `sentence-transformers` |
| Vector DB              | FAISS – Local vector index per Catan variant                             |
| LLM Backend            | [LLaMA 3 via Groq API](https://console.groq.com/docs) – used for answer generation |
| GitHub Repository      | [github.com/dewiri/ai-application-rag](https://github.com/dewiri/ai-application-rag) |

----


### Data Sources

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

## Chunking

### Method Used

To enable meaningful and performant document retrieval, the rulebooks were processed into manageable "chunks". Over time, the chunking strategy evolved to support variant-awareness, semantic filtering, and hybrid retrieval.

## Chunking Strategy

| Version | Method                              | Configuration                         | Description                                                                                                                                                                      |
|---------|--------------------------------------|----------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| V1.0    | Character-based                      | Fixed size only                        | Naive splitting into fixed-length chunks without considering sentence or paragraph boundaries. **Discarded** due to incoherent chunks that break mid-sentence.                  |
| V2.0    | SentenceTransformersTokenTextSplitter | Max 256 tokens                         | Token-based semantic splitting. **Not used in the final system** because it produced many overly short chunks, which are unsuitable for rule-heavy texts like board game manuals. |
| V3.0    | RecursiveCharacterTextSplitter       | 1000 characters, 100 character overlap | Paragraph-aware splitting that preserves semantic coherence. Well-suited for structured texts like rulebooks.                                                                  |
| V3.1    | Variant-Aware Chunk Mapping          | Based on filename keywords             | Automatically assigns documents to game variants (e.g., `basegame`, `seafarers`), enabling **modular vector store creation** and variant-specific retrieval.                    |
| V3.2    | Semantic Filtering + Metadata        | Embedding + contextual metadata        | Each chunk is stored with **variant metadata and embeddings**, enabling hybrid retrieval, score boosting, and transparent variant-aware search and evaluation.                 |

> **V3.0–V3.2** are combined in the current system to provide clean, modular, and context-aware chunking and retrieval.

---

## Vector Store

We use **FAISS** to store and retrieve dense embeddings for semantic search over Catan rule texts.

### V1 – Initial Setup

- Single **combined** FAISS index for all rulebooks  
- Embeddings generated with **OpenAI’s `text-embedding-ada-002`**
- No awareness of game variants or metadata

### V2 – Variant-Aware Indexing

- One FAISS index **per game variant** (e.g., `basegame`, `seafarers`, `cities_knights`, etc.)
- Uses V2.1 and V2.2 chunking methods for clean separation and flexibility
- Chunks include **semantic embedding** and **variant metadata**
- Enables:
  - Precise retrieval by variant
  - **Hybrid retrieval** (semantic + keyword scoring)
  - **Cross-variant similarity analysis**
  - Context-aware boosting and ranking logic

> This modular FAISS setup improves performance, precision, and explainability of retrieved answers.

---

## User Interface (Streamlit)

- Built using **Streamlit** for simplicity and interactivity.
- Core features include:

  - **Natural language input**: Users can type any rule-related question.
  - **LLM-powered answer generation**: Answers are generated based on retrieved rulebook content using LLaMA 3 via Groq.
  - **Expandable context view**: Retrieved rulebook chunks are displayed for full transparency.
  - **Hybrid retrieval toggle**: Option to enable/disable hybrid search (semantic + keyword-based).
  - **Cross-variant similarity**: Displays top-matching rule passages from *other* game variants to highlight possible differences.
  - **Score insights**: Dense, keyword, and hybrid scores are shown for each result when hybrid mode is active.

> The UI is optimized for transparency and explainability, making rule interpretation easier across game variants.
---

## Retrieval Methods: Before vs. Now

| Aspect                        | Before (Single-Variant Retrieval)                            | Now (Variant-Aware Retrieval + Cross-Variant Analysis)                  |
|-------------------------------|--------------------------------------------------------------|-------------------------------------------------------------------------|
| Vector Store Setup             | One single FAISS index for all game rules combined           | Separate FAISS index per game variant (basegame, seafarers, etc.)       |
| Query Execution               | Search performed on the combined index                        | Search primarily on selected variant’s index                            |
| Cross-Variant Comparison      | Not available                                                | Question also compared across all variant indexes for similarity        |
| Hybrid Retrieval              | Not implemented                                              | Combines semantic similarity and keyword matching within variant       |
| Result Precision             | Lower precision, generic answers                             | Higher precision, variant-specific answers                             |
| Transparency                 | No insight into variant differences                         | Shows similarity scores across all variants to highlight rule differences |
| User Experience              | Simpler but less accurate                                    | More informative and accurate, supports complex queries                 |


>Together, these upgrades lead to more accurate, explainable, and variant-specific answers — a major step beyond naive, single-index retrieval.

---

# Hybrid Retrieval Enhancement

## Overview

To improve the accuracy and relevance of retrieved rule passages for user queries, a **Hybrid Retrieval** method was implemented  that combines:

- **Dense Semantic Similarity** (using FAISS and vector embeddings)
- **Keyword-Based Matching** (using Jaccard similarity of query and chunk terms)

This approach balances semantic understanding with exact keyword overlap, leading to more precise and contextually relevant retrieval results.

---

## How It Works

### 1. Dense Retrieval (Semantic Similarity)

- Embeds the user query and document chunks using OpenAI’s `text-embedding-ada-002`.
- Uses FAISS to find chunks closest in vector space.
- Converts the L2 distance into a similarity score between 0 and 1.

### 2. Sparse Retrieval (Keyword Overlap)

- Extracts terms (words with ≥3 characters) from the query and each chunk.
- Computes Jaccard similarity (intersection over union) of term sets.

### 3. Hybrid Scoring

- Combines both scores using a weighted sum:
- HybridScore = α * DenseScore + (1 - α) * KeywordScore
- Typically, α = 0.8 (80% semantic, 20% keyword)

---

## Benefits

- **Captures semantic relationships** beyond exact word matches.
- **Mitigates irrelevant matches** by considering keyword presence.
- **Improves retrieval quality**, especially in a rule-based domain like Catan where keywords matter.

---

## Implementation Details

- The function `retrieve_hybrid(query, top_k, variant, alpha=0.8)` performs this retrieval.
- Fetches twice as many chunks from FAISS (top_k * 2) for better candidate pool.
- Returns the top_k chunks ranked by the hybrid score.
- Each chunk includes:
  - `chunk`: Text snippet
  - `hybrid_score`: Combined score
  - `dense_score`: Semantic similarity component
  - `keyword_score`: Keyword overlap component

---

## Integration

- Integrated into the Streamlit app with a checkbox to toggle hybrid retrieval.
- When enabled, users see retrieved chunks with detailed scoring for transparency.

---

## Example Scores from Real Queries

| Chunk | Hybrid Score | Dense Score | Keyword Score |
|-------|--------------|-------------|---------------|
| 1     | 0.389        | 0.617       | 0.047         |
| 2     | 0.384        | 0.615       | 0.037         |
| 3     | 0.376        | 0.600       | 0.039         |


### Score Interpretation

> A perfect score of `1.0` would mean full semantic + keyword match — very rare in practice.  
> In this context, **dense scores ≥ 0.60** and **keyword scores ~0.03–0.05** already indicate **strong retrieval quality**

---


## Evaluation & Testing

The chatbot was evaluated in two distinct stages:

---

### 1. Initial Evaluation (Single Vectorstore)

#### Method A: Keyword-Based Scoring (LLaMA 3 via Groq)

- **Dataset:** 40 real Catan-related questions
- **Approach:** Check if expected keyword appears in the model output
- **Result:**  
  25 out of 40 matched  
  → **Accuracy: 62.5%**

#### Method B: Manual Review (GPT-4o)

- **Approach:** Human-reviewed for overall correctness
- **Results:**

| Category           | Count | Percentage | Description                                  |
|--------------------|-------|------------|----------------------------------------------|
| Correct            | 31    | 77.5 %     | Fully or mostly accurate responses           |
| Partially correct  | 6     | 15.0 %     | Incomplete or slightly misleading            |
| Incorrect          | 3     | 7.5 %      | Factually wrong or unclear                   |

> **Conclusion:** Quality-based review reveals ~78% effective accuracy, better than the keyword-only method.

---

### 2. Updated Evaluation: Keyword-Based Scoring (LLaMA 3 via Groq)

- **Dataset:** 40 real Catan-related questions
- **Approach:** Check if expected keyword appears in the model output
- **Result:**  
  32 out of 40 matched  
  → **Accuracy: 80%**

#### Evaluation Results

| Category           | Count | Percentage | Description                                  |
|--------------------|-------|------------|----------------------------------------------|
| Correct            | 34    | 85.0 %     | Fully or mostly accurate responses           |
| Partially correct  | 4     | 10.0 %     | Incomplete or slightly misleading            |
| Incorrect          | 2     | 5.0 %      | Factually wrong or unclear                   |

> **Conclusion:**  The introduction of game variant-specific vectorstores significantly improved answer quality. Keyword match accuracy rose from **62.5% to 80%**, and manual review shows that **85%** of answers are now fully correct. This demonstrates the value of routing queries through variant-aware context retrieval for more accurate and relevant rule-based responses.

---

## Environment

- When running **locally**, environment variables (e.g. API keys) are loaded via `.env` using `python-dotenv`.
- On **Streamlit Cloud**, all credentials are securely managed via **Streamlit secrets** (`.streamlit/secrets.toml`).

---

## References

- [Catan Official Rulebooks](https://www.catan.com/service/game-rules)
- [FAISS: Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
- [Groq API Documentation](https://console.groq.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain – Text Splitters](https://docs.langchain.com/docs/components/text-splitters/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [PDFPlumber – Extract text from PDFs](https://github.com/jsvine/pdfplumber)