"""
Tools used by the CrewAI agents:

1. DocumentSearchTool  -> semantic search over an uploaded PDF's contents
                           (markitdown -> chonkie chunking -> fastembed
                           embeddings -> Qdrant vector store)
2. WebSearchTool        -> fallback search using Firecrawl, used only when
                           the PDF doesn't have a good answer
"""

import os
import uuid
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
from markitdown import MarkItDown
from chonkie import SemanticChunker


# --------------------------------------------------------------------------
# 1. PDF SEARCH TOOL
# --------------------------------------------------------------------------

class DocumentSearchInput(BaseModel):
    query: str = Field(..., description="The search query to look up in the uploaded document.")


class DocumentSearchTool(BaseTool):
    name: str = "document_search"
    description: str = (
        "Search the uploaded PDF document for information relevant to the query. "
        "Use this FIRST before trying anything else."
    )
    args_schema: Type[BaseModel] = DocumentSearchInput

    def __init__(self, file_path: str, collection_name: str = None, **kwargs):
        super().__init__(**kwargs)
        self._file_path = file_path
        self._collection_name = collection_name or f"pdf_{uuid.uuid4().hex[:8]}"
        self._embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        # In-memory Qdrant instance -- swap for QdrantClient(url="http://localhost:6333")
        # if you're running a persistent Qdrant server (recommended for real use).
        self._client = QdrantClient(":memory:")
        self._index_document()

    def _extract_markdown(self) -> str:
        md = MarkItDown()
        result = md.convert(self._file_path)
        return result.text_content

    def _index_document(self):
        text = self._extract_markdown()

        chunker = SemanticChunker(
            embedding_model="minishlab/potion-base-8M",
            threshold=0.5,
            chunk_size=512,
        )
        chunks = [c.text for c in chunker.chunk(text)]

        vectors = list(self._embedder.embed(chunks))
        dim = len(vectors[0])

        self._client.recreate_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )

        points = [
            models.PointStruct(id=i, vector=vectors[i].tolist(), payload={"text": chunks[i]})
            for i in range(len(chunks))
        ]
        self._client.upsert(collection_name=self._collection_name, points=points)

    def _run(self, query: str) -> str:
        query_vector = list(self._embedder.embed([query]))[0]
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector.tolist(),
            limit=5,
        )
        hits = response.points

        if not hits:
            return "NOT_FOUND: no relevant passages in the document."

        results = "\n\n---\n\n".join(h.payload["text"] for h in hits)
        return results


# --------------------------------------------------------------------------
# 2. WEB SEARCH FALLBACK TOOL (Firecrawl)
# --------------------------------------------------------------------------

class WebSearchInput(BaseModel):
    query: str = Field(..., description="The search query to look up on the web.")


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Search the live web for an answer. Only use this if document_search "
        "returned NOT_FOUND or clearly insufficient information."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str) -> str:
        try:
            from firecrawl import FirecrawlApp
        except ImportError:
            return "Firecrawl not installed. Run: pip install firecrawl-py"

        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            return "FIRECRAWL_API_KEY is not set. Add it to your .env file."

        app = FirecrawlApp(api_key=api_key)
        result = app.search(query, params={"limit": 3})

        snippets = []
        for item in result.get("data", []):
            snippets.append(f"{item.get('title', '')}: {item.get('description', '')}")
        return "\n".join(snippets) if snippets else "NOT_FOUND on the web either."