import logging
from typing import ClassVar, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "jinaai/jina-embeddings-v2-small-en"


class EmbeddingEncoder:
    """
    Generates dense semantic embeddings for candidate and job description text
    using a local Sentence Transformer model.

    The model is loaded once and cached at the class level so that multiple
    instances of EmbeddingEncoder share the same in-memory model.
    It supports loading from a local offline directory if available.
    """

    _model_cache: ClassVar[dict[str, SentenceTransformer]] = {}

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        """
        Args:
            model_name: The Sentence Transformer model identifier.
                        Defaults to BAAI/bge-small-en-v1.5.
        """
        self._model_name = model_name
        self._model = self._get_or_load_model(model_name)

    def encode_candidate(self, text: str) -> np.ndarray:
        """
        Encodes candidate profile text into a dense embedding vector.

        Args:
            text: Concatenated candidate profile text.

        Returns:
            A 1-D numpy array representing the embedding.
        """
        return self._encode(text)

    def encode_job_description(self, text: str) -> np.ndarray:
        """
        Encodes job description text into a dense embedding vector.

        Args:
            text: Concatenated job description text.

        Returns:
            A 1-D numpy array representing the embedding.
        """
        return self._encode(text)

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Encodes a batch of texts into dense embedding vectors.

        Args:
            texts: A list of text strings.

        Returns:
            A 2-D numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([])

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return np.asarray(embeddings)

    @property
    def embedding_dim(self) -> int:
        """Returns the dimensionality of the embedding vectors."""
        return self._model.get_embedding_dimension()

    def _encode(self, text: str) -> np.ndarray:
        """Encodes a single text string into a normalised embedding vector."""
        if not text or not text.strip():
            logger.warning("Empty text provided to encoder, returning zero vector.")
            return np.zeros(self.embedding_dim, dtype=np.float32)

        embedding = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embedding)

    @classmethod
    def _get_or_load_model(cls, model_name: str) -> SentenceTransformer:
        """Loads the model once and caches it at the class level."""
        import os
        
        if model_name not in cls._model_cache:
            # Check if the model has been downloaded locally
            local_model_path = os.path.join("data", "models", model_name.split("/")[-1])
            
            if os.path.exists(local_model_path):
                logger.info("Loading Sentence Transformer from local offline path: %s", local_model_path)
                cls._model_cache[model_name] = SentenceTransformer(
                    local_model_path,
                    device="cpu",
                    local_files_only=True,
                    trust_remote_code=False
                )
            else:
                logger.info("Loading Sentence Transformer from Hub: %s", model_name)
                cls._model_cache[model_name] = SentenceTransformer(
                    model_name,
                    device="cpu",
                    trust_remote_code=False
                )
            logger.info("Model loaded successfully: %s", model_name)
        return cls._model_cache[model_name]
