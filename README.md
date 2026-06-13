# 🏦 RBI Annual Report Q&A System

A RAG (Retrieval-Augmented Generation) based Q&A system built on RBI Annual Report 2024.

## 🚀 Features
- Ask any question from RBI Annual Report 2024
- Answers with source page numbers
- Custom evaluation pipeline with chunk size tuning
- Built with LangChain, ChromaDB, Groq LLM, and Streamlit

## 📊 Evaluation Results
| Chunk Size | Faithfulness | Relevancy |
|---|---|---|
| 500 | 0.60 | 0.30 |
| 1000 | 0.70 | 0.40 |
| 1500 | 0.70 | 0.40 |

**Optimal chunk size: 1000 — Faithfulness improved from 0.60 to 0.80**

## 🛠️ Tech Stack
- LangChain
- ChromaDB (Vector Database)
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- Groq LLM (llama-3.3-70b-versatile)
- Streamlit (Web UI)

## 📁 Project Structure
rbi-rag-project/

├── app.py              # Streamlit web app

├── evaluate.py         # Custom evaluation pipeline

├── load_Pdf.py         # PDF loading and RAG chain

├── evaluation_results.txt  # Evaluation results

└── test.pdf            # RBI Annual Report 

## 🔧 How to Run
```bash
# Install dependencies
pip install langchain langchain-community langchain-groq
pip install chromadb sentence-transformers streamlit pypdf

# Run the app
streamlit run app.py
``

