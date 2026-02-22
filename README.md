# 🩺 Clinical Protocol Assistant (RAG + Llama 3)

![Clinical Assistant Demo](assets/demo.png)

**A Production-Grade Microservice for Clinical Decision Support**  
Engineered with FastAPI, Docker, Qdrant, and Llama 3.

**Status:** 🟢 Active  
**Architecture:** Microservices  
**Deployment:** Dockerized  

---

## 📖 About This Project

This is not just a chatbot. It is a deterministic **Retrieval-Augmented Generation (RAG)** system designed to assist medical professionals by retrieving accurate, citation-backed answers from the *WHO Guidelines for the Pharmacological and Radiotherapeutic Management of Cancer Pain*.

Unlike standard LLMs that hallucinate, this system is engineered for **zero-trust verification**:

- **Strict Grounding:** Every answer is derived exclusively from the indexed vector database.
- **Citation Engine:** Each response includes the exact page number and supporting source text.
- **Microservice Architecture:** Decoupled FastAPI backend and Streamlit frontend.
- **Production-Ready Design:** Fully containerized using Docker Compose for scalable deployment.

---

## 🛠️ Tech Stack

- **AI/LLM:** Llama 3.3 (70B) via Groq  
- **Orchestration:** LlamaIndex  
- **Vector Database:** Qdrant (Dockerized)  
- **Backend:** FastAPI (Async Microservice)  
- **Frontend:** Streamlit  
- **Deployment:** Docker Compose  

---

## 📦 How to Run (Docker)

1. Clone the repository  
2. Create a `.env` file and add your `GROQ_API_KEY`  
3. Run:

```bash
docker-compose up --build
```
4. Access the UI at http://localhost:8501 and the API docs at http://localhost:8000/docs
.

..


