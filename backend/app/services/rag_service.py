from pathlib import Path
from typing import Dict
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # or langchain_text_splitters if on LC>=0.2
from chromadb import PersistentClient
from app.services.embedding_loader import EmbeddingModel
import hashlib

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
DOCS_PATH = BASE_DIR / "data" / "knowledge"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "supporting_docs"

embedding_model = EmbeddingModel()
client = PersistentClient(path=str(CHROMA_DIR))

def _get_or_create_collection():
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

def _chunk_id(source: str, idx: int) -> str:
    return hashlib.sha1(f"{source}::{idx}".encode("utf-8")).hexdigest()

def load_documents_from_directory() -> Dict[str, int]:
    """Parse PDF/MD/TXT, chunk, embed, and upsert to Chroma (no DB reset)."""
    loaders = [
        DirectoryLoader(str(DOCS_PATH), glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(str(DOCS_PATH), glob="**/*.txt", loader_cls=lambda p: TextLoader(p, encoding="utf-8")),
        DirectoryLoader(str(DOCS_PATH), glob="**/*.md",  loader_cls=lambda p: TextLoader(p, encoding="utf-8")),
    ]

    docs = []
    for loader in loaders:
        try:
            docs.extend(loader.load())
        except Exception as e:
            print(f"[INGEST] loader issue: {e}")

    if not docs:
        print("[INGEST] No documents found.")
        return {"chunks": 0}

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    if not chunks:
        print("[INGEST] No chunks produced.")
        return {"chunks": 0}

    texts, metadatas, ids = [], [], []
    per_source_counts = {}

    for c in chunks:
        src = c.metadata.get("source") or c.metadata.get("file_path") or "unknown"
        per_source_counts[src] = per_source_counts.get(src, 0) + 1
        idx = per_source_counts[src] - 1

        text = " ".join(c.page_content.split()) 
        texts.append(text)
        metadatas.append({"source": src, "chunk": idx})
        ids.append(_chunk_id(src, idx))

    embeddings = embedding_model.embed(texts)

    col = _get_or_create_collection()
    B = 128
    for s in range(0, len(texts), B):
        col.upsert(
            ids=ids[s:s+B],
            documents=texts[s:s+B],
            embeddings=embeddings[s:s+B],
            metadatas=metadatas[s:s+B],
        )

    print(f"Indexed {len(texts)} chunks into ChromaDB collection '{COLLECTION_NAME}'.")
    return {"chunks": len(texts)}

def query_rag(question: str, top_k: int = 3):
    """Return top docs + sources for a question."""
    col = _get_or_create_collection()
    q_vec = embedding_model.embed([question])[0]
    res = col.query(
        query_embeddings=[q_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    if not docs:
        return {"context": "", "sources": []}

    context = "\n\n".join(docs)
    sources = [{"source": m.get("source", ""), "chunk": m.get("chunk", -1)} for m in metas]
    return {"context": context, "sources": sources}
