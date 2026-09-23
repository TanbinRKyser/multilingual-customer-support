"""Dependency-free knowledge-base retrieval for the default application path."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import math
import re


BASE_DIR = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = BASE_DIR / "data" / "knowledge"
TOKEN_RE = re.compile(r"[^\W_]{2,}", re.UNICODE)


@dataclass(frozen=True)
class KnowledgeDocument:
    source: str
    text: str
    tokens: frozenset[str]


def _clean(text: str) -> str:
    text = re.sub(r"[#*_`]", "", text)
    return " ".join(text.split())


def _tokens(text: str) -> frozenset[str]:
    return frozenset(TOKEN_RE.findall(text.casefold()))


@lru_cache(maxsize=1)
def load_documents() -> tuple[KnowledgeDocument, ...]:
    documents: list[KnowledgeDocument] = []
    for path in sorted(KNOWLEDGE_DIR.glob("*")):
        if path.suffix.casefold() in {".txt", ".md"}:
            text = _clean(path.read_text(encoding="utf-8"))
        elif path.suffix.casefold() == ".pdf":
            try:
                from pypdf import PdfReader
                text = _clean(" ".join(page.extract_text() or "" for page in PdfReader(path).pages))
            except (ImportError, ModuleNotFoundError):
                continue
            except Exception:
                continue
        else:
            continue
        if text:
            documents.append(KnowledgeDocument(path.name, text, _tokens(text)))
    return tuple(documents)


def search_knowledge(query: str, limit: int = 3) -> list[tuple[KnowledgeDocument, float]]:
    query_tokens = _tokens(query)
    if not query_tokens:
        return []
    scored = []
    for document in load_documents():
        overlap = query_tokens & document.tokens
        if overlap:
            score = len(overlap) / math.sqrt(len(query_tokens) * len(document.tokens))
            scored.append((document, score))
    return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]
