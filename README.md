# SHL Assessment Recommendation System

A semantic search tool that recommends SHL assessments based on natural language job descriptions. Built with FAISS, Sentence Transformers, Gemini Pro, and deployed via FastAPI + Streamlit.
Webapp link : https://recommendationsystem123.streamlit.app/
---

## Features

- Semantic matching using `Sentence-Transformers`
- Fast top-10 similarity search using `FAISS` (cosine)
- Gemini Pro re-ranks results for better relevance
- FastAPI backend + Streamlit frontend (deployed)

---

## Approach

1. **Scraping**: Collected SHL assessments using BeautifulSoup
2. **Preprocessing**: Cleaned data, extracted duration, tagged “short/long test”
3. **Embedding**: Generated sentence embeddings using `all-MiniLM-L6-v2`
4. **Indexing**: Used FAISS (`IndexFlatIP`) for similarity search
5. **Re-ranking**: Top 10 results re-ordered using Gemini Pro
6. **Frontend & API**: Built with Streamlit + FastAPI, deployed on Render

---

## How to Set Up Locally

```bash
git clone https://github.com/your-username/shl-assessment-recommender.git
cd shl-assessment-recommender

# Install dependencies
pip install -r requirements.txt

# Generate embeddings & FAISS index
python build_index.py

# Run backend (localhost:8000)
uvicorn backend:app --reload

# Run frontend (opens in browser)
streamlit run frontend.py
```
