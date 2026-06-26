import logging
from dataclasses import dataclass
from typing import List

from backend.models import Candidate, JobDescription, CandidateFeatures
from backend.features.engine import FeatureEngine

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 100
DEFAULT_RELEVANCE_THRESHOLD = 0.15


@dataclass
class RetrievalResult:
    """Pairs a candidate with its computed features for downstream ranking."""
    candidate: Candidate
    features: CandidateFeatures
    relevance_score: float


class CandidateRetriever:
    """
    Retrieves the most relevant candidates from a pool before ranking.

    Retrieval is NOT ranking. This module narrows the candidate set using
    a lightweight relevance score derived from feature signals so that the
    ranking system operates on a manageable and already-relevant subset.
    """

    def __init__(
        self,
        feature_engine: FeatureEngine | None = None,
        top_k: int = DEFAULT_TOP_K,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    ):
        """
        Args:
            feature_engine: Engine used to compute candidate features.
                            A default instance is created if not provided.
            top_k: Maximum number of candidates to return.
            relevance_threshold: Minimum relevance score to include a candidate.
        """
        self._feature_engine = feature_engine or FeatureEngine()
        self._top_k = top_k
        self._relevance_threshold = relevance_threshold

    def retrieve(
        self,
        candidates: List[Candidate],
        jd: JobDescription,
    ) -> List[RetrievalResult]:
        """
        Retrieves the top candidate subset most relevant to a job description.

        Args:
            candidates: The full pool of candidates.
            jd: The structured job description.

        Returns:
            A list of RetrievalResult sorted by relevance score (descending),
            capped at top_k and filtered by the relevance threshold.
        """
        if not candidates:
            logger.warning("Empty candidate list provided to retriever.")
            return []

        scored_results: List[RetrievalResult] = []

        for candidate in candidates:
            try:
                features = self._feature_engine.extract_features(candidate, jd)
                relevance = self._compute_relevance(features)
                scored_results.append(
                    RetrievalResult(
                        candidate=candidate,
                        features=features,
                        relevance_score=relevance,
                    )
                )
            except Exception:
                logger.warning(
                    "Failed to compute features for candidate %s, skipping.",
                    candidate.candidate_id,
                    exc_info=True,
                )

        filtered = self._apply_threshold(scored_results)
        ranked = self._sort_by_relevance(filtered)
        top_results = ranked[: self._top_k]

        logger.info(
            "Retrieved %d candidates out of %d (threshold=%.2f, top_k=%d).",
            len(top_results),
            len(candidates),
            self._relevance_threshold,
            self._top_k,
        )

        return top_results

    def _compute_relevance(self, features: CandidateFeatures) -> float:
        """
        Computes a lightweight relevance score from feature signals.

        This is intentionally simpler than ranking. It uses a weighted sum
        of the core retrieval signals to quickly filter irrelevant candidates.
        """
        weights = {
            "technical_match_score": 0.30,
            "semantic_match_score": 0.25,
            "experience_score": 0.20,
            "validation_score": 0.10,
            "availability_score": 0.10,
            "behavioral_score": 0.05,
        }

        score = 0.0
        for field, weight in weights.items():
            value = getattr(features, field, None)
            if value is not None:
                score += value * weight

        return min(1.0, max(0.0, score))

    def _apply_threshold(
        self, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Filters out candidates below the relevance threshold."""
        return [r for r in results if r.relevance_score >= self._relevance_threshold]

    def _sort_by_relevance(
        self, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Sorts results by relevance score in descending order."""
        return sorted(results, key=lambda r: r.relevance_score, reverse=True)
