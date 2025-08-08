# test_rag.py
from app.services.rag_service import load_documents_from_directory, query_rag

def main():
    print("=== Testing RAG Pipeline ===")

    # Step 1: Ingest documents
    print("[1] Ingesting documents...")
    stats = load_documents_from_directory()
    print(f"Indexed {stats.get('chunks', stats)} chunks into ChromaDB.")

    # Step 2: Query the RAG store
    test_question = "How do I reset my password?"
    print(f"[2] Querying for: '{test_question}'")
    result = query_rag(test_question, top_k=3)

    # Step 3: Display context & sources
    print("\n=== Retrieved Context ===")
    print(result["context"])

    print("\n=== Sources ===")
    for src in result["sources"]:
        print(f" - {src['source']} (chunk {src['chunk']})")

    print("\n✅ RAG pipeline test complete.")

if __name__ == "__main__":
    main()
