import logging
from typing import Dict, List

from backend.models import Candidate, JobDescription
from backend.retrieval.retriever import RetrievalStrategy
from backend.embeddings.encoder import EmbeddingEncoder
from backend.embeddings.index import EmbeddingIndex

logger = logging.getLogger(__name__)


class EmbeddingStrategy(RetrievalStrategy):
    """
    Retrieval strategy using dense vector embeddings.
    Computes semantic similarity between pre-computed candidate
    embeddings (stored in a FAISS index) and a job description embedding.
    """

    def __init__(
        self,
        encoder: EmbeddingEncoder,
        index: EmbeddingIndex,
        top_k: int = 1000,
    ):
        """
        Initializes the EmbeddingStrategy.

        Args:
            encoder: The EmbeddingEncoder to use for encoding the job description.
            index: The populated EmbeddingIndex containing candidate embeddings.
            top_k: The maximum number of candidates to retrieve from the index.
        """
        self._encoder = encoder
        self._index = index
        self._top_k = top_k

    def score_candidates(
        self,
        candidates: List[Candidate],
        jd: JobDescription,
    ) -> Dict[str, float]:
        """
        Scores candidates by embedding the job description and querying the FAISS index.

        Args:
            candidates: The full pool of candidates (used to filter the results).
            jd: The structured job description.

        Returns:
            A dict mapping candidate_id to a normalized semantic similarity score in [0, 1].
        """
        if not candidates:
            return {}

        jd_text = self._format_jd_text(jd)
        query_embedding = self._encoder.encode_job_description(jd_text)

        # Search for up to top_k candidates in the index
        search_results = self._index.search(query_embedding, top_k=self._top_k)

        # Build a fast lookup for requested candidates
        requested_ids = {c.candidate_id for c in candidates}

        scores: Dict[str, float] = {}
        for cand_id, raw_score in search_results:
            if cand_id in requested_ids:
                # Normalize raw cosine similarity/inner product [-1, 1] to [0, 1]
                normalized_score = self._normalize_score(raw_score)
                scores[cand_id] = normalized_score

        return scores

    def _format_jd_text(self, jd: JobDescription) -> str:
        """
        Formats a JobDescription object into a concatenated string suitable for embedding.
        """
        parts = [jd.title]

        if jd.responsibilities:
            parts.extend(jd.responsibilities)

        if jd.must_have_skills:
            parts.extend(jd.must_have_skills)

        if jd.preferred_skills:
            parts.extend(jd.preferred_skills)

        if jd.required_domains:
            parts.extend(jd.required_domains)

        if jd.keywords:
            parts.extend(jd.keywords)

        if jd.behavioral_expectations:
            parts.extend(jd.behavioral_expectations)

        if jd.culture_preferences:
            parts.extend(jd.culture_preferences)

        return " ".join(p for p in parts if p)

    def _normalize_score(self, score: float) -> float:
        """
        Normalizes a cosine similarity score from [-1, 1] to [0, 1].
        """
        normalized = (score + 1.0) / 2.0
        return max(0.0, min(1.0, float(normalized)))
