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

| Type                          | Configuration                         |
|-------------------------------|----------------------------------------|
| RecursiveCharacterTextSplitter | 1000 characters, 100 character overlap |


---

## Vector Store

- Used **FAISS** to store precomputed embeddings
- Embeddings were created once using **OpenAI's `text-embedding-ada-002`**
- Stored to disk and reused at runtime

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

---

## Environment

Environment variables managed with `python-dotenv`:


## References

- [Catan Official Rulebooks](https://www.catan.com/service/game-rules)
- [FAISS: Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
- [Groq API Documentation](https://console.groq.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain – Text Splitters](https://docs.langchain.com/docs/components/text-splitters/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [PDFPlumber – Extract text from PDFs](https://github.com/jsvine/pdfplumber)