import logging
from typing import Dict, List, Optional

from backend.retrieval.retriever import RetrievalResult

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "hybrid_retrieval_score": 0.35,
    "technical_match_score": 0.20,
    "semantic_match_score": 0.10,
    "experience_score": 0.10,
    "recruiter_interest_score": 0.05,
    "behavioral_score": 0.05,
    "availability_score": 0.05,
    "validation_score": 0.05,
    "culture_fit_score": 0.05,
}

class RankingEngine:
    """
    Ranks candidates retrieved by the HybridRetriever.
    
    Computes a deterministic final score using a weighted combination
    of the hybrid retrieval score and independent signals from CandidateFeatures.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initializes the RankingEngine.
        
        Args:
            weights: Optional dictionary of weights for different signals.
                     If a signal is not provided, its weight defaults to 0.0.
                     The default weights sum to 1.0.
        """
        self._weights = weights if weights is not None else DEFAULT_WEIGHTS
        self._normalize_weights()
        
    def _normalize_weights(self) -> None:
        """Normalizes the configured weights so that they sum to 1.0."""
        total = sum(w for w in self._weights.values() if w > 0)
        if total <= 0:
            logger.warning("RankingEngine initialized with zero or negative total weights.")
            return
            
        self._weights = {k: max(0.0, v) / total for k, v in self._weights.items()}

    def rank(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Ranks a list of RetrievalResult objects.
        
        Computes the final_score for each candidate, stores it in their
        features, and returns the sorted list.
        
        Args:
            results: A list of candidate results from the retrieval pipeline.
            
        Returns:
            A list of RetrievalResult objects sorted by final_score (descending).
        """
        if not results:
            return []
            
        for result in results:
            final_score = self._compute_final_score(result)
            result.features.final_score = final_score
            
        # Sort by final_score in descending order
        sorted_results = sorted(
            results,
            key=lambda r: r.features.final_score if r.features.final_score is not None else 0.0,
            reverse=True
        )
        
        logger.info("Successfully ranked %d candidates.", len(sorted_results))
        return sorted_results
        
    def _compute_final_score(self, result: RetrievalResult) -> float:
        """
        Computes the weighted final score for a single candidate.
        """
        score = 0.0
        
        # Add the hybrid retrieval score component
        hybrid_weight = self._weights.get("hybrid_retrieval_score", 0.0)
        score += result.relevance_score * hybrid_weight
        
        # Add the feature score components
        for feature_name, weight in self._weights.items():
            if feature_name == "hybrid_retrieval_score":
                continue
                
            # Safely fetch the feature value; treat None as 0.0
            feature_val = getattr(result.features, feature_name, 0.0)
            if feature_val is None:
                feature_val = 0.0
                
            score += feature_val * weight
            
        return min(1.0, max(0.0, score))
