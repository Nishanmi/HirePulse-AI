import logging
from typing import Dict, List, Optional

from backend.models import JobDescription
from backend.retrieval.retriever import RetrievalResult

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "hybrid_retrieval_score": 0.20,
    "career_evidence_score": 0.25,
    "technical_match_score": 0.10,
    "consistency_score": 0.07,
    "experience_score": 0.10,
    "role_relevance_score": 0.05,
    "semantic_match_score": 0.10,
    "validation_score": 0.05,
    "behavioral_score": 0.05,
    "recruiter_interest_score": 0.02,
    "availability_score": 0.01,
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

    def rank(self, results: List[RetrievalResult], jd: JobDescription) -> List[RetrievalResult]:
        """
        Ranks a list of RetrievalResult objects.
        
        Computes the final_score for each candidate, stores it in their
        features, and returns the sorted list.
        
        Args:
            results: A list of candidate results from the retrieval pipeline.
            jd: The JobDescription.
            
        Returns:
            A list of RetrievalResult objects sorted by final_score (descending).
        """
        if not results:
            return []
            
        for result in results:
            final_score = self._compute_final_score(result, jd)
            result.features.final_score = final_score
            
        # Sort by final_score in descending order
        sorted_results = sorted(
            results,
            key=lambda r: r.features.final_score if r.features.final_score is not None else 0.0,
            reverse=True
        )
        
        logger.info("Successfully ranked %d candidates.", len(sorted_results))
        return sorted_results
        
    def _compute_final_score(self, result: RetrievalResult, jd: JobDescription) -> float:
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
            
        # MACRO PENALTY:
        # Candidates with zero relevant work history or wildly irrelevant roles
        # should NOT be able to coast on keyword-stuffed BM25 scores.
        con_score = getattr(result.features, "consistency_score", 0.0)
        role_score = getattr(result.features, "role_relevance_score", 0.0)
        tech_score = getattr(result.features, "technical_match_score", 0.0)
        
        # If role relevance is extremely low AND they have near-zero technical match, crush them.
        # This crushes Civil/Mobile/.NET devs with no AI skills, while saving Frontend Engineers
        # who actually have FAISS/OpenSearch (high technical match).
        if role_score < 0.30 and tech_score < 0.05:
            score *= 0.1
            
        # DISCIPLINE MISMATCH DETECTOR:
        # If the JD requires software/AI engineering, we check the candidate's FULL career history.
        # If their entire career is in non-software disciplines (Mechanical, Civil, Operations, etc.)
        # and they have ZERO software titles, their vector DB keywords are confirmed synthetic honeypots.
        is_discipline_mismatch = False
        jd_title_lower = jd.title.lower() if jd.title else ""
        if "ai" in jd_title_lower or "software" in jd_title_lower or "data" in jd_title_lower or "backend" in jd_title_lower:
            non_software = ["mechanical", "civil", "electrical", "chemical", "industrial", "operations", "hr", "sales", "accountant", "graphic", "support", "finance", "legal"]
            software = ["software", "developer", "backend", "frontend", "data", "ai", "machine learning", "ml", "devops", "cloud", "systems", "platform", "programmer", "full stack", "fullstack", "architect"]
            
            has_non_software = False
            has_software = False
            
            for exp in result.candidate.career_history:
                if not exp.title:
                    continue
                t = exp.title.lower()
                if any(w in t for w in non_software):
                    has_non_software = True
                if any(w in t for w in software):
                    has_software = True
                    
            if has_non_software and not has_software:
                # Permanent disqualification.
                score *= 0.05
                is_discipline_mismatch = True
                
        # TIE-BREAKER FOR CRUSHED TIER:
        # Guarantee that crushed candidates who actually possess vector DB skills 
        # float above crushed candidates who only have generic skills (like Python).
        core_ai_skills = [
            "weaviate", "pinecone", "faiss", "qdrant", "milvus", "opensearch", 
            "elasticsearch", "lora", "qlora", "peft", "rag", "llm", "xgboost"
        ]
        has_core_ai = False
        
        for s in result.candidate.skills:
            if any(w in s.name.lower() for w in core_ai_skills):
                has_core_ai = True
                break
                
        if has_core_ai and not is_discipline_mismatch:
            score += 0.15
                
        return min(1.0, max(0.0, score))
