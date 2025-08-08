# embedding_model.py
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        :param model_name: HuggingFace sentence-transformers model.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts, batch_size: int = 32):
        """
        :param texts: str or list[str]
        :param batch_size: controls memory usage
        :return: list of embeddings
        """
        if isinstance( texts, str ):
            texts = [ texts ]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        return embeddings.tolist()
