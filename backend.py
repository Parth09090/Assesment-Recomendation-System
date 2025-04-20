import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import nest_asyncio
import uvicorn
import pandas as pd
# Allow FastAPI to run in Jupyter's event loop
nest_asyncio.apply()

# Initialize FastAPI app
app = FastAPI()

# Load the FAISS index and embeddings which we have created in other file here 
index = faiss.read_index("shl_index.faiss")
embeddings = np.load("shl_embeddings.npy")

# Loading the SentenceTransformer model all-mpnet-base-v2 
model = SentenceTransformer('all-mpnet-base-v2')

df = pd.read_csv("test_data.xls")
assessment_data = df.to_dict(orient="records")
import re, ast

def rerank_with_gemini(query, faiss_results, k=10):
    """
    query:                the user’s natural‑language query
    faiss_results:        list of dicts from FAISS search (each dict has keys
                          'name','test_type','duration','remote_testing','adaptive','description')
    k:                    number of top items to return (max 10, min 1)
    """
    # prompt
    prompt = f"""
You are an intelligent recruitment assistant.

Given the following user query:
"{query}"

And the list of assessments below, return the **top {k} most relevant** by their number,
in this format: [3, 1, 2, 5, 4]

Assessments:
"""
    for i, test in enumerate(faiss_results):
        prompt += f"""
#{i+1}
Name: {test['name']}
Type: {test['test_type']}
Duration: {test['duration']}
Remote: {test['remote_testing']}
Adaptive: {test['adaptive']}
Description: {test['description']}
"""

    prompt += f"\nOnly return the top {k} numbers in a Python-style list like: [3, 2, 1, 4, 5]"

    # Calling  Gemini
    try:
        response = model_gemini.generate_content(prompt)
        text = response.text.strip()
        print("Gemini response:", text)

        #Trying safe literal eval first
        try:
            lst = ast.literal_eval(text)
            indices = [i for i in lst if isinstance(i, int)]
        except Exception:
            # if Fallback: split on any non-digit
            parts = re.split(r'\D+', text)
            indices = [int(p) for p in parts if p.isdigit()]

        #Convert to zero-based, if filter out-of-range
        sel = [i-1 for i in indices]
        sel = [i for i in sel if 0 <= i < len(faiss_results)]

    except Exception as e:
        print("Gemini error:", e)
        # Fallback to the original FAISS ordering
        sel = list(range(len(faiss_results)))

    # Clamp to between 1 and k
    n = min(max(len(sel), 1), k)

    #Returning the reranked items
    return [faiss_results[i] for i in sel[:n]]

# request body structure for /recommend endpoint
class RecommendRequest(BaseModel):
    query: str

# Define response model
class ProductRecommendation(BaseModel):
    name: str
    url: str
    remote_testing: str
    adaptive: str
    duration: str
    test_type: str

# /recommend endpoint
@app.post("/recommend")
async def recommend(request: RecommendRequest):
    query = request.query
    # Generate embedding
    query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')
    # Ensuring the  correct shape accoriding to our model (1, 768)
    query_embedding = np.array(query_embedding).reshape(1, -1)
    # Search for tje FAISS index
    D, I = index.search(query_embedding, 10)

    # top 10 recommended tests
    recommendations = []
    for idx in I[0]:
        row = df.iloc[idx]
        recommendation = {
            "name": row["Test_Name"],
            "url": "https://www.shl.com"+row["Test_Link"],
            "remote_testing": row["Remote_Testing"],
            "adaptive": row["Adaptive_Testing"],
            "duration": row["Assessment_Length"],
            "description":row["Description"],
            "test_type": row["Test_Types"]
        }
        recommendations.append(recommendation)



    # Re-rank using Gemini
    reranked_recommendations = rerank_with_gemini(query, recommendations)
    return {"recommendations": reranked_recommendations}

@app.get("/health")
async def health():
    return {"status": "OK"}

