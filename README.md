# Catan Rule Expert Bot (RAG)

---

### 📘 Project Description

This is a **Retrieval-Augmented Generation (RAG)** application that acts as an expert assistant for the board game **Catan**, including the **base game** and all major **expansions**.

Users can ask natural-language questions like:

> _"Can I build a city next to my own road?"_

The system retrieves relevant rule passages and uses a language model to generate **accurate, context-based answers**.

---

### Repository & Demo

| Name                      | URL                                  |
|---------------------------|---------------------------------------|
| Streamlit App             | https://catan-rulebot.streamlit.app   |
| Embedding Model (OpenAI)  | https://platform.openai.com/docs     |
| Code (GitHub Repository)  | https://github.com/dewiri/ai-application-rag |

---

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

| Version | Type                          | Configuration                         | Description                                                                 |
|---------|-------------------------------|----------------------------------------|-----------------------------------------------------------------------------|
| 1       | Character-based               | Fixed size only                        | Naive splitting into chunks of fixed length without regard to semantics.   |
| 2       | RecursiveCharacterTextSplitter | 1000 characters, 100 character overlap | Splits on paragraphs/sentences first (`\n\n`, `\n`, `.`, `!`, `?`, etc.), leading to more natural chunks. |


---

## Vector Store

We use **FAISS** to store precomputed embeddings, enabling fast semantic search over rule texts.

### Version 1 (Initial Setup)

- One **single** FAISS vector store for all rulebooks combined  
- Embeddings created with **OpenAI’s `text-embedding-ada-002`**  
- Chunks stored once and reused at runtime  

### Version 2 (Updated – Variant-Aware)

- One **dedicated FAISS vector store per game variant**  
  (e.g., `basegame`, `seafarers`, `cities_knights`, etc.)  
- Embeddings still created using **OpenAI’s `text-embedding-ada-002`**  
- Stored **separately per rulebook**, enabling more precise, variant-specific retrieval and similarity scoring  
- Allows **cross-variant comparisons**  
  (e.g., “how does this rule differ in Seafarers vs. Basegame?”)  

> This upgrade enables a more accurate, modular, and explainable retrieval process.

---

## LLMs Used

| Name         | Access via | Usage                             |
|--------------|------------|------------------------------------|
| LLaMA 3 (70B) | Groq API   | Used for answering user questions |

---

## UI

- Built with **Streamlit**
- Includes:
  - Input field for user question
  - LLM model
  - Display of generated answer
  - Expandable section for retrieved context
  - Similarity analysis across all game variants
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

---

**Summary:**  
The current system uses variant-specific vector stores with optional cross-variant similarity analysis, enabling more precise and transparent rule retrieval tailored to the selected game variant.

---

## Environment

- When running **locally**, environment variables (e.g. API keys) are loaded via `.env` using `python-dotenv`.
- On **Streamlit Cloud**, all credentials are securely managed via **Streamlit secrets** (`.streamlit/secrets.toml`).


## Evaluation & Testing

The chatbot was evaluated in two distinct stages:

---

### 1. Initial Evaluation (Single Vectorstore)

#### 🔍 Method A: Keyword-Based Scoring (LLaMA 3 via Groq)

- **Dataset:** 40 real Catan-related questions
- **Approach:** Check if expected keyword appears in the model output
- **Result:**  
  25 out of 40 matched  
  → **Accuracy: 62.5%**

#### 🔍 Method B: Manual Review (GPT-4o)

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

> **Conclusion:**  
The introduction of game variant-specific vectorstores significantly improved answer quality. Keyword match accuracy rose from **62.5% to 80%**, and manual review shows that **85%** of answers are now fully correct. This demonstrates the value of routing queries through variant-aware context retrieval for more accurate and relevant rule-based responses.


## References

- [Catan Official Rulebooks](https://www.catan.com/service/game-rules)
- [FAISS: Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
- [Groq API Documentation](https://console.groq.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain – Text Splitters](https://docs.langchain.com/docs/components/text-splitters/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [PDFPlumber – Extract text from PDFs](https://github.com/jsvine/pdfplumber)