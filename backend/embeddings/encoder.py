import logging
import os
from typing import ClassVar, Optional

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# Suppress harmless C++ device discovery warnings from ONNX
ort.set_default_logger_severity(3)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "jinaai/jina-embeddings-v2-small-en"

class EmbeddingEncoder:
    """
    Generates dense semantic embeddings for candidate and job description text
    using an ONNX Runtime model.

    The model and tokenizer are loaded once and cached at the class level so that multiple
    instances of EmbeddingEncoder share the same in-memory model.
    It supports loading from a local offline directory if available.
    """

    _session_cache: ClassVar[dict[str, ort.InferenceSession]] = {}
    _tokenizer_cache: ClassVar[dict[str, AutoTokenizer]] = {}

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        """
        Args:
            model_name: The Sentence Transformer model identifier.
                        Defaults to jinaai/jina-embeddings-v2-small-en.
        """
        self._model_name = model_name
        self._tokenizer = self._get_or_load_tokenizer(model_name)
        self._session = self._get_or_load_session(model_name)

    def encode_candidate(self, text: str) -> np.ndarray:
        """
        Encodes candidate profile text into a dense embedding vector.
        """
        return self._encode(text)

    def encode_job_description(self, text: str) -> np.ndarray:
        """
        Encodes job description text into a dense embedding vector.
        """
        return self._encode(text)

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Encodes a batch of texts into dense embedding vectors.
        """
        if not texts:
            return np.array([])
            
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            tokens = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="np"
            )
            
            onnx_inputs = {
                "input_ids": tokens["input_ids"].astype(np.int64),
                "attention_mask": tokens["attention_mask"].astype(np.int64),
            }
            if "token_type_ids" in tokens:
                onnx_inputs["token_type_ids"] = tokens["token_type_ids"].astype(np.int64)
                
            outputs = self._session.run(None, onnx_inputs)
            embeddings = outputs[0]
            
            # Apply L2 Normalization (essential for cosine similarity)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings_normalized = embeddings / np.maximum(norms, 1e-12)
            all_embeddings.append(embeddings_normalized)
            
        return np.vstack(all_embeddings)

    @property
    def embedding_dim(self) -> int:
        """Returns the dimensionality of the embedding vectors."""
        return 512

    def _encode(self, text: str) -> np.ndarray:
        """Encodes a single text string into a normalised embedding vector."""
        if not text or not text.strip():
            logger.warning("Empty text provided to encoder, returning zero vector.")
            return np.zeros(self.embedding_dim, dtype=np.float32)

        return self.encode_batch([text], batch_size=1)[0]

    @classmethod
    def _get_or_load_tokenizer(cls, model_name: str) -> AutoTokenizer:
        if model_name not in cls._tokenizer_cache:
            local_model_path = os.path.join("data", "models", model_name.split("/")[-1])
            if os.path.exists(local_model_path):
                cls._tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
            else:
                cls._tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name)
        return cls._tokenizer_cache[model_name]

    @classmethod
    def _get_or_load_session(cls, model_name: str) -> ort.InferenceSession:
        if model_name not in cls._session_cache:
            local_model_path = os.path.join("data", "models", model_name.split("/")[-1])
            onnx_path = os.path.join(local_model_path, "model-w-mean-pooling.onnx")
            
            if os.path.exists(onnx_path):
                logger.info("Loading ONNX Model from local offline path: %s", onnx_path)
                cls._session_cache[model_name] = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            else:
                raise FileNotFoundError(f"ONNX model not found at {onnx_path}")
        return cls._session_cache[model_name]
