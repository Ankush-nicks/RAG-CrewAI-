import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from crew_setup import build_crew

load_dotenv()

st.set_page_config(page_title="Agentic RAG", page_icon="📄", layout="wide")
st.title("📄 Agentic RAG — Chat with your PDF (+ web fallback)")

# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.header("Setup")

    model_choice = st.selectbox("LLM (runs locally via Ollama)", ["qwen2.5", "deepseek-r1"])

    uploaded_file = st.file_uploader("Upload a PDF for the knowledge base", type=["pdf"])

    if uploaded_file and st.button("Index document", type="primary"):
        with st.spinner("Reading, chunking, and embedding your PDF..."):
            tmp_dir = tempfile.mkdtemp()
            file_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.session_state.crew = build_crew(file_path, model_choice)
            st.session_state.file_name = uploaded_file.name
        st.success(f"Indexed '{uploaded_file.name}'. Ask away!")

    if "file_name" in st.session_state:
        st.caption(f"Active document: **{st.session_state.file_name}**")

    st.divider()
    st.caption(
        "Requires:\n"
        "- `ollama serve` running locally\n"
        "- `ollama pull llama3.2` and/or `ollama pull deepseek-r1`\n"
        "- `FIRECRAWL_API_KEY` in a `.env` file (for web fallback)"
    )

# ------------------------------------------------------------------ chat --
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question about your document...")

if query:
    if "crew" not in st.session_state:
        st.warning("Upload and index a PDF first (see sidebar).")
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and synthesizing..."):
                result = st.session_state.crew.kickoff(inputs={"query": query})
                answer = str(result)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})