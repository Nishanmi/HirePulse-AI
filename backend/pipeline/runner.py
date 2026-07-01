import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import faiss

from backend.candidate.parser import parse_candidates
from backend.jd.extractor import JDExtractor
from backend.embeddings.encoder import EmbeddingEncoder
from backend.embeddings.index import EmbeddingIndex
from backend.retrieval.bm25 import BM25Strategy
from backend.retrieval.embedding import EmbeddingStrategy
from backend.retrieval.hybrid import HybridRetriever
from backend.features.engine import FeatureEngine
from backend.retrieval.retriever import FeatureBasedStrategy, RetrievalResult
from backend.ranking.engine import RankingEngine
from backend.explainability.explanation_engine import ExplanationEngine
from backend.submission.exporter import SubmissionExporter

logger = logging.getLogger(__name__)

class PreloadedBM25Strategy(BM25Strategy):
    """Overrides BM25Strategy to use a precomputed BM25Okapi index."""
    def __init__(self, bm25_index):
        super().__init__()
        self.bm25_index = bm25_index

    def score_candidates(self, candidates, jd):
        if not candidates:
            return {}
        query = self._build_query(jd)
        if not query:
            logger.warning("BM25Strategy: empty query derived from job description.")
            return {c.candidate_id: 0.0 for c in candidates}
        tokenised_query = self._tokenise(query)
        raw_scores = self.bm25_index.get_scores(tokenised_query)
        scores = self._normalise_scores(raw_scores)
        return {
            candidates[i].candidate_id: scores[i]
            for i in range(len(candidates))
        }

class PipelineRunner:
    """
    Orchestrates the complete HirePulse end-to-end pipeline.
    
    This runner executes the full lifecycle: parsing candidates and job descriptions,
    building retrieval indexes, running hybrid retrieval, executing the ranking engine,
    generating deterministic explanations, and finally exporting the results
    to a valid CSV format.
    """

    def __init__(self, top_k_retrieval: int = 100):
        """
        Initializes the PipelineRunner.
        
        Args:
            top_k_retrieval: The maximum number of candidates to retrieve and rank.
                             Must be exactly 100 to meet the submission validator requirements.
        """
        self.top_k = top_k_retrieval
            
    def run(
        self, 
        dataset_path: str, 
        jd_text: str, 
        output_csv_path: str,
        index_dir: Optional[str] = None
    ) -> None:
        """
        Executes the full ranking pipeline.
        
        Args:
            dataset_path: Path to the JSON file containing the candidate dataset.
            jd_text: The raw text of the job description.
            output_csv_path: The file path where the final submission CSV will be saved.
            index_dir: Optional path to directory containing precomputed indexes.
        """
        logger.info("Starting HirePulse pipeline execution...")
        
        load_from_index = False
        if index_dir:
            idx_path = Path(index_dir)
            faiss_path = idx_path / "faiss.index"
            bm25_path = idx_path / "bm25.pkl"
            meta_path = idx_path / "candidate_metadata.jsonl"
            faiss_map_path = idx_path / "faiss_map.pkl"
            
            if faiss_path.exists() and bm25_path.exists() and meta_path.exists() and faiss_map_path.exists():
                load_from_index = True
                logger.info("Precomputed artifacts found in %s. Loading them...", index_dir)
            else:
                logger.info("Precomputed artifacts missing in %s. Falling back to rebuild.", index_dir)

        if load_from_index:
            logger.info("Loading candidates from candidate_metadata.jsonl...")
            candidates = []
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        candidates.append(Candidate.model_validate_json(line))
            logger.info("Successfully loaded %d candidates from precomputed metadata.", len(candidates))
        else:
            # 1. Parse candidates
            logger.info("Loading candidates from %s...", dataset_path)
            try:
                candidates = parse_candidates(dataset_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load candidates from {dataset_path}: {e}") from e
                
            if not candidates:
                raise ValueError(f"No valid candidates parsed from {dataset_path}")
            logger.info("Successfully loaded %d candidates.", len(candidates))

        # 2. Extract JD
        logger.info("Extracting structured job description...")
        try:
            extractor = JDExtractor()
            jd = extractor.extract(jd_text)
        except Exception as e:
            raise RuntimeError(f"Failed to extract job description: {e}") from e
        logger.info("Extracted Job Description for role: %s", jd.title)

        # 3. Build/Load Embeddings
        logger.info("Initializing Embedding Encoder...")
        try:
            encoder = EmbeddingEncoder()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedding encoder: {e}") from e
            
        # 3a. Encode JD Role Title for Role Similarity
        logger.info("Encoding Job Description role title...")
        jd.normalized_role_title = FeatureEngine.normalize_role_title(jd.title)
        if jd.normalized_role_title:
            jd_role_emb = encoder.encode_batch([jd.normalized_role_title], max_length=16)[0]
            jd.role_embedding = jd_role_emb.tolist()

        # 3b. Encode JD Requirements Text for Career Description Evidence
        logger.info("Encoding Job Description requirements text...")
        jd_req_parts = [jd.title]
        jd_req_parts.extend(jd.responsibilities)
        jd_req_text = " ".join(filter(None, jd_req_parts))
        if jd_req_text.strip():
            jd_req_emb = encoder.encode_batch([jd_req_text], max_length=256)[0]
            jd.jd_requirements_embedding = jd_req_emb.tolist()
            
        index = EmbeddingIndex(embedding_dim=encoder.embedding_dim)
        
        if load_from_index:
            logger.info("Loading FAISS index...")
            index._index = faiss.read_index(str(faiss_path))
            
            logger.info("Loading FAISS ID map...")
            with open(faiss_map_path, "rb") as f:
                index._candidate_map = pickle.load(f)
                
            if index._candidate_map:
                index._next_id = max(index._candidate_map.keys()) + 1
            else:
                index._next_id = 0
            logger.info("FAISS index loaded successfully.")
        else:
            logger.info("Encoding candidate texts and building FAISS index...")
            candidate_ids = [c.candidate_id for c in candidates]
            candidate_texts = [self._extract_candidate_text(c) for c in candidates]
            
            try:
                embeddings = encoder.encode_batch(candidate_texts)
                index.build(embeddings, candidate_ids)
            except Exception as e:
                raise RuntimeError(f"Failed to build embedding index: {e}") from e
            logger.info("Successfully built FAISS index.")

        # 4. Retrieval
        logger.info("Initializing retrieval strategies...")
        feature_strategy = FeatureBasedStrategy()
        
        if load_from_index:
            logger.info("Loading BM25 index...")
            with open(bm25_path, "rb") as f:
                bm25_index = pickle.load(f)
            bm25_strategy = PreloadedBM25Strategy(bm25_index)
        else:
            bm25_strategy = BM25Strategy()
            
        # We fetch extra candidates during embedding strategy to ensure sufficient recall
        embedding_strategy = EmbeddingStrategy(encoder=encoder, index=index, top_k=self.top_k * 3)
        
        hybrid_retriever = HybridRetriever(
            strategies=[
                (feature_strategy, 1.0),
                (bm25_strategy, 1.5),
                (embedding_strategy, 2.0),
            ],
            top_k=self.top_k,
            relevance_threshold=0.005 # Low threshold to ensure we hit 100 for small datasets
        )
        
        logger.info("Running Hybrid Retrieval to fetch top %d candidates...", self.top_k)
        retrieved_results = hybrid_retriever.retrieve(candidates, jd)
        
        if len(retrieved_results) < self.top_k:
            logger.warning(
                "Retrieved only %d candidates. Submission expects exactly %d. "
                "Ensure your dataset is large enough.", 
                len(retrieved_results), self.top_k
            )

        # 5. Ranking
        logger.info("Running Ranking Engine...")
        ranking_engine = RankingEngine()
        ranked_results = ranking_engine.rank(retrieved_results, jd)
        
        # Ensure strict secondary sorting (Score DESC, Candidate ID ASC)
        logger.info("Applying strict validation sorting...")
        sorted_results = sorted(
            ranked_results, 
            key=lambda r: (
                -(r.features.final_score if r.features.final_score is not None else 0.0), 
                r.candidate.candidate_id
            )
        )

        # 5b. Debug Breakdown for Top 10
        logger.info("=========================================")
        logger.info("         TOP 10 DEBUG BREAKDOWN          ")
        logger.info("=========================================")
        for i, result in enumerate(sorted_results[:10], start=1):
            feats = result.features
            logger.info(
                "Rank %d | %s | Title: %s",
                i, result.candidate.candidate_id, result.candidate.profile.current_title
            )
            logger.info(
                "  -> Hybrid: %.3f | Role: %.3f | Career: %.3f | Tech: %.3f | Sem: %.3f | Exp: %.3f | Signals: %.3f | FINAL: %.3f",
                result.relevance_score,
                getattr(feats, 'role_relevance_score', 0) or 0,
                getattr(feats, 'career_evidence_score', 0) or 0,
                getattr(feats, 'technical_match_score', 0) or 0,
                getattr(feats, 'semantic_match_score', 0) or 0,
                getattr(feats, 'experience_score', 0) or 0,
                getattr(feats, 'recruiter_interest_score', 0) or 0,
                getattr(feats, 'final_score', 0) or 0
            )
        logger.info("=========================================")

        # 6. Explanations
        logger.info("Generating explanations for the Top 100 candidates...")
        explanation_engine = ExplanationEngine()
        
        # SLICE TO TOP 100 HERE to satisfy CSV constraints
        final_top_100 = sorted_results[:100]
        
        export_items: List[Tuple[RetrievalResult, str]] = []
        for result in final_top_100:
            try:
                reasoning = explanation_engine.explain(result.candidate, result.features, jd)
                export_items.append((result, reasoning))
            except Exception as e:
                logger.error("Failed to generate explanation for %s: %s", result.candidate.candidate_id, e)
                raise RuntimeError(f"Explanation generation failed: {e}") from e

        # 7. Export
        logger.info("Exporting %d candidates to %s...", len(export_items), output_csv_path)
        exporter = SubmissionExporter()
        try:
            exporter.export(export_items, output_csv_path)
        except Exception as e:
            raise RuntimeError(f"Failed to export submission CSV: {e}") from e
            
        logger.info("Pipeline execution completed successfully!")

    def _extract_candidate_text(self, candidate) -> str:
        """Helper to extract and format candidate text for embeddings."""
        parts = [
            candidate.profile.headline,
            candidate.profile.summary,
            candidate.profile.current_title,
            candidate.profile.current_company,
            candidate.profile.current_industry,
        ]
        for role in candidate.career_history:
            parts.extend([role.title, role.company, role.industry, role.description])
        for edu in candidate.education:
            parts.extend([edu.institution, edu.degree, edu.field_of_study])
        for skill in candidate.skills:
            parts.append(skill.name)
            
        return " ".join(filter(None, parts))
