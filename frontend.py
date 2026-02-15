import streamlit as st
import requests
import os 

# CONFIG
# If we are in Docker, use the service name 'backend'. If local, use 'localhost'.
BACKEND_HOST = os.getenv("BACKEND_HOST", "localhost")
API_URL = f"http://{BACKEND_HOST}:8000/ask"

st.set_page_config(page_title="Clinical AI Assistant", page_icon="🩺", layout="centered")
# HEADER
st.title("🩺 Clinical Protocol Assistant")
st.caption("Powered by Llama 3 (Groq) & RAG • Verified WHO Guidelines")

# SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []

# DISPLAY HISTORY
for message in st.session_state.messages:
    # This automatically handles the styling based on the config.toml now!
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# CHAT INPUT
if prompt := st.chat_input("Ask about dosage, protocols, or contraindications..."):
    # 1. Show User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get Bot Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing Clinical Guidelines..."):
            try:
                response = requests.post(API_URL, json={"question": prompt})
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]
                    
                    st.markdown(answer)
                    
                    # Sources in a clean expander
                    with st.expander("📚 Verified Sources"):
                        for idx, source in enumerate(sources):
                            st.info(f"**Page {source['page']}**: {source['text']}")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                
                else:
                    st.error(f"Error {response.status_code}")

            except Exception as e:
                st.error(f"Connection Error: {e}")