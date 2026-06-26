import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Union

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingIndex:
    """
    Manages a FAISS vector index for fast similarity search of candidate embeddings.
    
    This class wraps FAISS to handle string-based candidate IDs and provides
    persistence capabilities for the index and ID mapping.
    """

    def __init__(self, embedding_dim: int = 384):
        """
        Initialises the EmbeddingIndex.

        Args:
            embedding_dim: The dimensionality of the dense vectors (default 384 for bge-small).
        """
        self._embedding_dim = embedding_dim
        # IndexFlatIP uses Inner Product (equivalent to Cosine Similarity for normalised vectors)
        base_index = faiss.IndexFlatIP(embedding_dim)
        # IndexIDMap allows us to associate arbitrary integer IDs with vectors
        self._index = faiss.IndexIDMap(base_index)
        
        self._candidate_map: Dict[int, str] = {}
        self._next_id = 0

    def build(self, embeddings: np.ndarray, candidate_ids: List[str]) -> None:
        """
        Builds or adds to the FAISS index with candidate embeddings.

        Args:
            embeddings: A 2D numpy array of shape (N, embedding_dim).
            candidate_ids: A list of N string candidate IDs corresponding to the embeddings.
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self._embedding_dim:
            raise ValueError(f"Embeddings must be a 2D array with shape (N, {self._embedding_dim})")
        if embeddings.shape[0] != len(candidate_ids):
            raise ValueError("Number of embeddings must match number of candidate IDs")
        if len(candidate_ids) == 0:
            logger.warning("Empty embeddings array provided to build.")
            return

        # FAISS strictly requires C-contiguous float32 arrays
        embeddings_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Generate integer IDs for FAISS since it doesn't support strings natively
        num_new = len(candidate_ids)
        faiss_ids = np.arange(self._next_id, self._next_id + num_new, dtype=np.int64)
        
        # Store string mapping
        for i, cid in zip(faiss_ids, candidate_ids):
            self._candidate_map[int(i)] = cid

        # Add to index
        self._index.add_with_ids(embeddings_f32, faiss_ids)
        self._next_id += num_new
        
        logger.info(f"Added {num_new} embeddings to FAISS index. Total: {self._index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """
        Searches the index for the most similar candidates to the query.

        Args:
            query_embedding: A 1D or 2D numpy array containing the query vector.
            top_k: The number of results to return.

        Returns:
            A list of (candidate_id, similarity_score) tuples, sorted by similarity descending.
        """
        if self._index.ntotal == 0:
            return []

        query_f32 = np.ascontiguousarray(query_embedding, dtype=np.float32)
        if query_f32.ndim == 1:
            query_f32 = query_f32.reshape(1, -1)

        if query_f32.shape[1] != self._embedding_dim:
            raise ValueError(f"Query embedding must have dimension {self._embedding_dim}")

        k = min(top_k, self._index.ntotal)
        
        # scores and indices are arrays of shape (1, k)
        scores, indices = self._index.search(query_f32, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # FAISS returns -1 for missing neighbors if k > ntotal
                candidate_id = self._candidate_map[int(idx)]
                results.append((candidate_id, float(score)))

        return results

    def save(self, path: Union[str, Path]) -> None:
        """
        Saves the FAISS index and the candidate ID mapping to disk.

        Args:
            path: Directory path where the index files will be saved.
        """
        base_path = Path(path)
        base_path.mkdir(parents=True, exist_ok=True)

        index_file = base_path / "index.faiss"
        map_file = base_path / "candidate_map.json"
        meta_file = base_path / "meta.json"

        # Write index
        faiss.write_index(self._index, str(index_file))

        # Write map
        with open(map_file, "w", encoding="utf-8") as f:
            json.dump(self._candidate_map, f)

        # Write metadata
        meta = {
            "embedding_dim": self._embedding_dim,
            "next_id": self._next_id,
            "ntotal": self._index.ntotal
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        logger.info(f"Saved FAISS index with {self._index.ntotal} vectors to {base_path}")

    def load(self, path: Union[str, Path]) -> None:
        """
        Loads the FAISS index and the candidate ID mapping from disk.

        Args:
            path: Directory path where the index files were saved.
        """
        base_path = Path(path)
        index_file = base_path / "index.faiss"
        map_file = base_path / "candidate_map.json"
        meta_file = base_path / "meta.json"

        if not index_file.exists() or not map_file.exists() or not meta_file.exists():
            raise FileNotFoundError(f"Index files not found in {base_path}")

        # Load metadata
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
            self._embedding_dim = meta["embedding_dim"]
            self._next_id = meta["next_id"]

        # Load map (JSON keys are strings, must convert to int)
        with open(map_file, "r", encoding="utf-8") as f:
            str_map = json.load(f)
            self._candidate_map = {int(k): v for k, v in str_map.items()}

        # Load index
        self._index = faiss.read_index(str(index_file))

        logger.info(f"Loaded FAISS index with {self._index.ntotal} vectors from {base_path}")
