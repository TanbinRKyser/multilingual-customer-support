# test_embedding.py
def main():
    # Experimental model smoke test; intentionally excluded from normal tests.
    import numpy as np
    from app.services.embedding_loader import EmbeddingModel
    # Initialize
    emb_model = EmbeddingModel()

    # Test sentences
    texts = [
        "Hello world",
        "Wie geht es dir?",
        "I want to reset my password",
        "Track my order please"
    ]

    # Generate embeddings
    vectors = emb_model.embed(texts)

    # Print some info
    print(f"Number of embeddings: {len(vectors)}")
    print(f"Embedding dimension: {len(vectors[0])}")
    print("First vector snippet:", np.array(vectors[0])[:5])  # show first 5 dims

    # Sanity check: vectors should be list of lists
    assert isinstance(vectors, list)
    assert isinstance(vectors[0], list)
    print("✅ EmbeddingModel test passed.")

if __name__ == "__main__":
    main()
