import os
import qdrant_client
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.groq import Groq

# 1. INIT APP & CONFIG
app = FastAPI(title="Clinical RAG Agent API", version="1.0")
load_dotenv()

# Check keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing from .env")

# 2. SETUP THE BRAIN (Global Scope for Performance)
# We load this ONCE when the server starts, not for every request.
print("--- BOOTING AI KERNEL ---")
Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

try:
    client = qdrant_client.QdrantClient(path="./qdrant_db")
    vector_store = QdrantVectorStore(client=client, collection_name="clinical_guidelines")
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    # Create the query engine once
    query_engine = index.as_query_engine(similarity_top_k=5)
    print("--- AI KERNEL READY ---")
except Exception as e:
    print(f"CRITICAL ERROR BOOTING AI: {e}")
    query_engine = None

# 3. DEFINE DATA MODELS (Pydantic)
class QueryRequest(BaseModel):
    question: str

class Source(BaseModel):
    text: str
    page: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]

# 4. API ENDPOINTS
@app.get("/")
def health_check():
    return {"status": "active", "model": "llama-3.3-70b"}

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    if not query_engine:
        raise HTTPException(status_code=500, detail="AI Engine not initialized")
    
    try:
        # The Inference Step
        response = query_engine.query(request.question)
        
        # Parse Sources
        source_list = []
        for node in response.source_nodes:
            source_list.append(Source(
                text=node.text[:200] + "...", # Truncate for clean JSON
                page=node.metadata.get('page_label', 'N/A')
            ))
            
        return QueryResponse(answer=str(response), sources=source_list)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run on localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)