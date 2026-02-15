import os
import qdrant_client
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.groq import Groq  

# 1. LOAD SECRETS
load_dotenv()

# Check for Groq Key
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("CRITICAL: GROQ_API_KEY not found in .env file!")

# 2. SETUP THE BRAIN (Now using Llama 3 on Groq)
print("Initializing Llama 3 Model via Groq...")
Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

# 3. CONNECT TO YOUR DATA
print("Loading Knowledge Base from './qdrant_db'...")
client = qdrant_client.QdrantClient(path="./qdrant_db")
vector_store = QdrantVectorStore(client=client, collection_name="clinical_guidelines")

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# 4. CREATE THE ENGINE
chat_engine = index.as_chat_engine(
    chat_mode="context",
    similarity_top_k=5,
    system_prompt=(
        "You are a Clinical Guidelines Assistant. "
        "Answer strictly based on the provided context. "
        "If the answer is not in the context, say 'I cannot find that in the guidelines'. "
        "Always cite the page number if available."
    )
)

# 5. START CHATTING
print("\n" + "="*50)
print("CLINICAL RAG AGENT READY (Powered by Llama 3)")
print("Try asking: 'What is the conversion ratio for Oral Morphine to Hydromorphone?'")
print("="*50 + "\n")

while True:
    user_input = input("User: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    
    try:
        response = chat_engine.chat(user_input)
        print(f"\nAgent: {response}\n")
        
        print("-" * 20)
        print("Sources used:")
        for node in response.source_nodes:
            page = node.metadata.get('page_label', 'N/A')
            # Clean up newlines for cleaner display
            text_snippet = node.text[:100].replace('\n', ' ')
            print(f"• [Page {page}] {text_snippet}...")
        print("-" * 20 + "\n")
        
    except Exception as e:
        print(f"Error: {e}")