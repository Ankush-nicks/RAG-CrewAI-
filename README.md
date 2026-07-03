# Agentic RAG (CrewAI + Qdrant + Ollama)

A chatbot that answers questions from a PDF you upload, and automatically
falls back to a live web search when the PDF doesn't have the answer.
Built as a working recreation of `patchy631/ai-engineering-hub/agentic_rag`.

## How it works

1. You upload a PDF in the sidebar.
2. `markitdown` converts it to markdown text.
3. `chonkie`'s semantic chunker splits it into meaning-based chunks.
4. `fastembed` embeds each chunk; embeddings are stored in an in-memory **Qdrant** collection.
5. You ask a question in the chat box.
6. **Retriever Agent** searches the PDF first (`document_search` tool). If it
   finds nothing useful, it automatically tries `web_search` (Firecrawl).
7. **Response Synthesizer Agent** turns whatever was retrieved into a final answer.
8. Both agents run on a local LLM (Llama 3.2 or DeepSeek-R1) served by Ollama
   — no data leaves your machine except the optional web search call.

## Setup

### 1. Install Python deps (Python 3.11+)
```bash
pip install -r requirements.txt
```

### 2. Install and run Ollama
```bash
# https://ollama.com/download
ollama pull llama3.2
ollama pull deepseek-r1
ollama serve
```

### 3. (Optional) Set up web-search fallback
```bash
cp .env.example .env
# put your free Firecrawl API key (https://www.firecrawl.dev/) into .env
```

### 4. Run the app
```bash
streamlit run app.py
```

## Getting a PDF for the knowledge base

There's no fixed knowledge base — **any PDF works**. Options:

- Use your own document: a manual, contract, report, textbook chapter, notes, etc.
- Export any Word doc / Google Doc / webpage to PDF ("Print → Save as PDF").
- Grab a free PDF online, e.g.:
  - [arXiv.org](https://arxiv.org) — research papers
  - [Project Gutenberg](https://www.gutenberg.org) — free books
  - A company's public whitepaper or documentation PDF

Just upload it in the sidebar, click **Index document**, and start asking questions.

## Notes / things to customize

- The Qdrant client here is in-memory (`:memory:`) — it resets each time you
  restart the app. For persistence, run a real Qdrant server
  (`docker run -p 6333:6333 qdrant/qdrant`) and change
  `QdrantClient(":memory:")` to `QdrantClient(url="http://localhost:6333")`
  in `rag_tools.py`.
- Swap `BAAI/bge-small-en-v1.5` for a different embedding model in
  `rag_tools.py` if you want higher quality (bigger/slower) embeddings.
- You can add more file types by extending `DocumentSearchTool` — `markitdown`
  also supports Word, PowerPoint, and HTML out of the box.
