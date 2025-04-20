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

