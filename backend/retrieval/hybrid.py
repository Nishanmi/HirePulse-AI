import logging
from typing import Dict, List, Tuple

from backend.features.engine import FeatureEngine
from backend.models import Candidate, CandidateFeatures, JobDescription
from backend.retrieval.retriever import RetrievalResult, RetrievalStrategy

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 100
DEFAULT_RELEVANCE_THRESHOLD = 0.10


class HybridRetriever:
    """
    Combines multiple RetrievalStrategy implementations into a single
    retrieval pipeline using weighted score fusion.

    Each strategy independently scores the candidate pool. The scores are
    merged using configurable weights and normalised to produce a single
    hybrid relevance score per candidate. Results are filtered by a minimum
    threshold and capped at top_k.

    Retrieval is NOT ranking. This module narrows the candidate set so that
    the downstream ranking system operates on a relevant subset.
    """

    def __init__(
        self,
        strategies: List[Tuple[RetrievalStrategy, float]],
        feature_engine: FeatureEngine | None = None,
        top_k: int = DEFAULT_TOP_K,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    ):
        """
        Initialises the HybridRetriever.

        Args:
            strategies: A list of (strategy, weight) tuples. Each strategy's
                        scores are scaled by its weight before fusion.
                        Weights do not need to sum to 1; they are normalised
                        internally.
            feature_engine: Engine used to compute CandidateFeatures for each
                            retrieved candidate. A default instance is created
                            if not provided.
            top_k: Maximum number of candidates to return.
            relevance_threshold: Minimum hybrid score required for a candidate
                                 to be included in the results.
        """
        if not strategies:
            raise ValueError("At least one (strategy, weight) pair is required.")

        self._strategies = strategies
        self._feature_engine = feature_engine or FeatureEngine()
        self._top_k = top_k
        self._relevance_threshold = relevance_threshold

    def retrieve(
        self,
        candidates: List[Candidate],
        jd: JobDescription,
    ) -> List[RetrievalResult]:
        """
        Retrieves the top candidate subset most relevant to a job description
        by fusing scores from all configured strategies.

        Args:
            candidates: The full pool of candidates.
            jd: The structured job description.

        Returns:
            A list of RetrievalResult sorted by hybrid relevance score
            (descending), capped at top_k and filtered by the relevance
            threshold.
        """
        if not candidates:
            logger.warning("HybridRetriever: empty candidate list provided.")
            return []

        strategy_scores = self._collect_strategy_scores(candidates, jd)
        merged_scores = self._merge_scores(strategy_scores)
        candidate_map = {c.candidate_id: c for c in candidates}

        results = self._build_results(merged_scores, candidate_map, jd)
        results = self._sort_by_relevance(results)
        top_results = results[: self._top_k]

        logger.info(
            "HybridRetriever: retrieved %d candidates out of %d "
            "(threshold=%.2f, top_k=%d, strategies=%d).",
            len(top_results),
            len(candidates),
            self._relevance_threshold,
            self._top_k,
            len(self._strategies),
        )

        return top_results

    def _collect_strategy_scores(
        self,
        candidates: List[Candidate],
        jd: JobDescription,
    ) -> List[Dict[str, float]]:
        """Runs each strategy and collects its candidate score mapping."""
        all_scores: List[Dict[str, float]] = []
        for strategy, _ in self._strategies:
            try:
                scores = strategy.score_candidates(candidates, jd)
                all_scores.append(scores)
            except Exception:
                logger.warning(
                    "HybridRetriever: strategy %s failed, returning empty scores.",
                    type(strategy).__name__,
                    exc_info=True,
                )
                all_scores.append({})
        return all_scores

    def _merge_scores(
        self,
        strategy_scores: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Merges relevance scores from multiple strategies using weighted averaging.

        Each strategy's scores are multiplied by its corresponding weight.
        The sum is normalised by the total weight to produce a final merged
        score in [0, 1] for every candidate that appears in at least one
        strategy's output.
        """
        total_weight = sum(w for _, w in self._strategies)
        if total_weight == 0:
            return {}

        merged: Dict[str, float] = {}

        for scores, (_, weight) in zip(strategy_scores, self._strategies):
            for cand_id, score in scores.items():
                merged[cand_id] = merged.get(cand_id, 0.0) + (score * weight)

        for cand_id in merged:
            merged[cand_id] = max(0.0, min(1.0, merged[cand_id] / total_weight))

        return merged

    def _build_results(
        self,
        merged_scores: Dict[str, float],
        candidate_map: Dict[str, Candidate],
        jd: JobDescription,
    ) -> List[RetrievalResult]:
        """
        Constructs RetrievalResult objects for candidates that pass the
        relevance threshold. Features are computed for each qualifying
        candidate via the FeatureEngine.
        """
        results: List[RetrievalResult] = []

        for cand_id, relevance in merged_scores.items():
            if relevance < self._relevance_threshold:
                continue

            candidate = candidate_map.get(cand_id)
            if candidate is None:
                logger.warning(
                    "HybridRetriever: candidate %s in scores but not in pool, skipping.",
                    cand_id,
                )
                continue

            try:
                features = self._feature_engine.extract_features(candidate, jd)
            except Exception:
                logger.warning(
                    "HybridRetriever: failed to compute features for candidate %s, skipping.",
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

        return results

    def _sort_by_relevance(
        self, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Sorts results by relevance score in descending order."""
        return sorted(results, key=lambda r: r.relevance_score, reverse=True)
