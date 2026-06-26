import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

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


class RetrievalStrategy(ABC):
    """
    Abstract base class for a retrieval strategy.

    Each strategy produces a mapping of candidate_id -> relevance score
    for a given candidate pool and job description. Multiple strategies
    can be combined by CandidateRetriever to form a hybrid retrieval pipeline.
    """

    @abstractmethod
    def score_candidates(
        self,
        candidates: List[Candidate],
        jd: JobDescription,
    ) -> Dict[str, float]:
        """
        Scores each candidate for relevance to a job description.

        Args:
            candidates: The full pool of candidates.
            jd: The structured job description.

        Returns:
            A dict mapping candidate_id to a relevance score in [0, 1].
        """


class FeatureBasedStrategy(RetrievalStrategy):
    """
    Retrieval strategy that uses the FeatureEngine to compute a lightweight
    relevance score from deterministic feature signals.
    """

    _SIGNAL_WEIGHTS = {
        "technical_match_score": 0.30,
        "semantic_match_score": 0.25,
        "experience_score": 0.20,
        "validation_score": 0.10,
        "availability_score": 0.10,
        "behavioral_score": 0.05,
    }

    def __init__(self, feature_engine: FeatureEngine | None = None):
        self._feature_engine = feature_engine or FeatureEngine()

    def score_candidates(
        self,
        candidates: List[Candidate],
        jd: JobDescription,
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for candidate in candidates:
            try:
                features = self._feature_engine.extract_features(candidate, jd)
                scores[candidate.candidate_id] = self._compute_relevance(features)
            except Exception:
                logger.warning(
                    "FeatureBasedStrategy: failed to score candidate %s, skipping.",
                    candidate.candidate_id,
                    exc_info=True,
                )
        return scores

    def _compute_relevance(self, features: CandidateFeatures) -> float:
        score = 0.0
        for field, weight in self._SIGNAL_WEIGHTS.items():
            value = getattr(features, field, None)
            if value is not None:
                score += value * weight
        return min(1.0, max(0.0, score))


# ---------------------------------------------------------------------------
# Extension points for future strategies
# ---------------------------------------------------------------------------
#
# class BM25Strategy(RetrievalStrategy):
#     """
#     Retrieval strategy using BM25 term-frequency scoring.
#     Scores candidates by matching tokenised JD text against
#     candidate profile text using an inverted index.
#     """
#     def score_candidates(self, candidates, jd):
#         raise NotImplementedError
#
#
# class EmbeddingStrategy(RetrievalStrategy):
#     """
#     Retrieval strategy using dense vector embeddings.
#     Computes cosine similarity between pre-computed candidate
#     embeddings and a JD embedding.
#     """
#     def score_candidates(self, candidates, jd):
#         raise NotImplementedError
# ---------------------------------------------------------------------------


class CandidateRetriever:
    """
    Retrieves the most relevant candidates from a pool before ranking.

    Retrieval is NOT ranking. This module narrows the candidate set using
    one or more pluggable retrieval strategies so that the ranking system
    operates on a manageable and already-relevant subset.

    Multiple strategies are combined via weighted score merging (hybrid retrieval).
    """

    def __init__(
        self,
        strategies: List[tuple[RetrievalStrategy, float]] | None = None,
        feature_engine: FeatureEngine | None = None,
        top_k: int = DEFAULT_TOP_K,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    ):
        """
        Args:
            strategies: A list of (strategy, weight) tuples. Weights are used
                        to merge scores from each strategy. If not provided,
                        a single FeatureBasedStrategy with weight 1.0 is used.
            feature_engine: Engine passed to the default FeatureBasedStrategy.
                            Ignored if strategies are provided explicitly.
            top_k: Maximum number of candidates to return.
            relevance_threshold: Minimum merged relevance score to include a candidate.
        """
        if strategies:
            self._strategies = strategies
        else:
            self._strategies = [(FeatureBasedStrategy(feature_engine), 1.0)]

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

        # Collect scores from each strategy
        strategy_scores: List[Dict[str, float]] = []
        for strategy, _ in self._strategies:
            strategy_scores.append(strategy.score_candidates(candidates, jd))

        # Merge scores across strategies
        merged_scores = self._merge_scores(strategy_scores)

        # Build candidate lookup for fast access
        candidate_map = {c.candidate_id: c for c in candidates}

        # Compute features and build results
        results: List[RetrievalResult] = []
        for cand_id, relevance in merged_scores.items():
            if relevance < self._relevance_threshold:
                continue
            candidate = candidate_map[cand_id]
            try:
                features = self._feature_engine.extract_features(candidate, jd)
            except Exception:
                logger.warning(
                    "Failed to compute features for candidate %s, skipping.",
                    cand_id,
                    exc_info=True,
                )
                continue
            results.append(
                RetrievalResult(
                    candidate=candidate,
                    features=features,
                    relevance_score=relevance,
                )
            )

        results = self._sort_by_relevance(results)
        top_results = results[: self._top_k]

        logger.info(
            "Retrieved %d candidates out of %d (threshold=%.2f, top_k=%d, strategies=%d).",
            len(top_results),
            len(candidates),
            self._relevance_threshold,
            self._top_k,
            len(self._strategies),
        )

        return top_results

    def _merge_scores(
        self, strategy_scores: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Merges relevance scores from multiple strategies using weighted averaging.

        Each strategy's scores are multiplied by its weight and then normalised
        by the total weight to produce a final merged score per candidate.
        """
        total_weight = sum(w for _, w in self._strategies)
        if total_weight == 0:
            return {}

        merged: Dict[str, float] = {}

        for scores, (_, weight) in zip(strategy_scores, self._strategies):
            for cand_id, score in scores.items():
                merged[cand_id] = merged.get(cand_id, 0.0) + (score * weight)

        # Normalise by total weight
        for cand_id in merged:
            merged[cand_id] = min(1.0, max(0.0, merged[cand_id] / total_weight))

        return merged

    def _sort_by_relevance(
        self, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Sorts results by relevance score in descending order."""
        return sorted(results, key=lambda r: r.relevance_score, reverse=True)
