from typing import List, Dict, Any
import threading
from chromadb.api.types import Embeddings, Documents
import numpy as np
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

class MTSentenceTransformerEmbeddingFunction(SentenceTransformerEmbeddingFunction):
    # Since we do dynamic imports we have to type this as Any
    models: Dict[str, Any] = {}
    _lock = threading.Lock()

    # If you have a beefier machine, try "gtr-t5-large".
    # for a full list of options: https://huggingface.co/sentence-transformers, https://www.sbert.net/docs/pretrained_models.html
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize_embeddings: bool = False,
        **kwargs: Any,
    ):
        """Initialize SentenceTransformerEmbeddingFunction.

        Args:
            model_name (str, optional): Identifier of the SentenceTransformer model, defaults to "all-MiniLM-L6-v2"
            device (str, optional): Device used for computation, defaults to "cpu"
            normalize_embeddings (bool, optional): Whether to normalize returned vectors, defaults to False
            **kwargs: Additional arguments to pass to the SentenceTransformer model.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ValueError(
                "The sentence_transformers python package is not installed. Please install it with `pip install sentence_transformers`"
            )

        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        for key, value in kwargs.items():
            if not isinstance(value, (str, int, float, bool, list, dict, tuple)):
                raise ValueError(f"Keyword argument {key} is not a primitive type")
        self.kwargs = kwargs

        with self.__class__._lock:
            if model_name not in self.__class__.models:
                self.__class__.models[model_name] = SentenceTransformer(
                    model_name_or_path=model_name, device=device, **kwargs
                )
        self._model = self.models[model_name]
    
    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for the given documents.

        Args:
            input: Documents to generate embeddings for.

        Returns:
            Embeddings for the documents.
        """
        embeddings = self._model.encode(
            list(input),
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )

        return [np.array(embedding, dtype=np.float32) for embedding in embeddings]