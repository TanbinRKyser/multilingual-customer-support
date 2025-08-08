import os
from pathlib import Path
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
from app.services.embedding_loader import EmbeddingModel

CHROMA_DIR = "backend/chroma_db"
COLLECTION_NAME = "supporting_docs"
DOCS_PATH = "backend/data/knowledge"

embedding_model = EmbeddingModel()  # consider a multilingual model in that class
client = chromadb.Client(Settings(persist_directory=CHROMA_DIR, chroma_db_impl="duckdb+parquet"))

def _get_or_create_collection():
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        return client.create_collection(COLLECTION_NAME)

def load_documents_from_directory() -> Dict[str, int]:
    """Parse PDF/MD/TXT, chunk, embed, and persist to Chroma (no DB reset)."""
    loaders = [
        DirectoryLoader(DOCS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(DOCS_PATH, glob="**/*.txt", loader_cls=lambda p: TextLoader(p, encoding="utf-8")),
        DirectoryLoader(DOCS_PATH, glob="**/*.md",  loader_cls=lambda p: TextLoader(p, encoding="utf-8")),
    ]

    # Load
    docs = []
    for loader in loaders:
        try:
            docs.extend(loader.load())
        except Exception as e:
            print(f"[INGEST] loader issue: {e}")

    if not docs:
        return {"chunks": 0}

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Prepare payloads
    texts = [c.page_content for c in chunks]
    metadatas = []
    ids = []
    for i, c in enumerate(chunks):
        src = c.metadata.get("source") or c.metadata.get("file_path") or "unknown"
        metadatas.append({"source": src, "chunk": i})
        ids.append(f"{src}::chunk_{i}")  # deterministic id per source+chunk

    # Embed (batch)
    embeddings = embedding_model.embed(texts)

    # Upsert to collection
    col = _get_or_create_collection()
    # Add in batches to avoid large payloads
    B = 128
    for s in range(0, len(texts), B):
        col.upsert(
            ids=ids[s:s+B],
            documents=texts[s:s+B],
            embeddings=embeddings[s:s+B],
            metadatas=metadatas[s:s+B],
        )

    client.persist()
    print(f"Indexed {len(texts)} chunks into ChromaDB collection '{COLLECTION_NAME}'.")
    return {"chunks": len(texts)}


def query_rag(question: str, top_k: int = 3):
    """Return top docs + sources; you can feed this to FLAN-T5 later."""
    col = _get_or_create_collection()
    query_embedding = embedding_model.embed([question])[0]
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    # Build a simple joined context and return sources
    context = "\n\n".join(docs)
    sources = [{"source": m.get("source", ""), "chunk": m.get("chunk", -1)} for m in metas]
    return {"context": context, "sources": sources}