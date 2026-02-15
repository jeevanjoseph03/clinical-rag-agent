import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
import qdrant_client

# --- CONCEPT 1: Embeddings ---
# We don't send raw text to the DB. We send "Embeddings" (vectors of numbers).
# FastEmbed is a quantization-ready library that runs locally.
# It maps words with similar meanings close together in vector space.
print("Initializing Embeddings Model...")
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

# --- CONCEPT 2: Vector Database ---
# A standard SQL DB (like PostgreSQL) is bad at "semantic" search.
# Qdrant is optimized for high-dimensional vector math.
# We create a local client here (in production, this points to a Cloud URL).
print("Connecting to Vector DB...")
client = qdrant_client.QdrantClient(path="./qdrant_db") 
vector_store = QdrantVectorStore(client=client, collection_name="clinical_guidelines")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# --- CONCEPT 3: Chunking & Ingestion ---
# You can't feed a 100-page PDF to an LLM at once (Context Window limits).
# We MUST "chunk" it. LlamaIndex handles the splitting (default is 1024 tokens).
def ingest_data():
    # Create a 'data' folder and put a PDF there before running!
    if not os.path.exists('data'):
        os.makedirs('data')
        print("ALERT: Created 'data' folder. Please put a medical PDF inside it.")
        return

    print("Loading documents...")
    documents = SimpleDirectoryReader("./data").load_data()
    
    print(f"Loaded {len(documents)} document pages. Indexing...")
    
    # This acts as the "ETL" pipeline:
    # 1. Split text -> 2. Embed text -> 3. Store in Qdrant
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
    
    print("✅ Ingestion Complete! Data is now vector-searchable.")
    
    # Let's test it immediately (Sanity Check)
    # This uses "Vector Search" (Cosine Similarity), not an LLM yet.
    query_engine = index.as_query_engine()
    response = query_engine.query("What is the summary of this document?")
    print(f"\nTest Response: {response}")

if __name__ == "__main__":
    ingest_data()