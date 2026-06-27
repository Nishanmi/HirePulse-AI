import logging
from typing import List, Tuple

from backend.candidate.parser import parse_candidates
from backend.jd.extractor import JDExtractor
from backend.embeddings.encoder import EmbeddingEncoder
from backend.embeddings.index import EmbeddingIndex
from backend.retrieval.bm25 import BM25Strategy
from backend.retrieval.embedding import EmbeddingStrategy
from backend.retrieval.hybrid import HybridRetriever
from backend.retrieval.retriever import FeatureBasedStrategy, RetrievalResult
from backend.ranking.engine import RankingEngine
from backend.explainability.explanation_engine import ExplanationEngine
from backend.submission.exporter import SubmissionExporter

logger = logging.getLogger(__name__)

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
        if top_k_retrieval != 100:
            logger.warning("Submission requires exactly 100 candidates. Overriding top_k to 100.")
            self.top_k = 100
        else:
            self.top_k = top_k_retrieval
            
    def run(self, dataset_path: str, jd_text: str, output_csv_path: str) -> None:
        """
        Executes the full ranking pipeline.
        
        Args:
            dataset_path: Path to the JSON file containing the candidate dataset.
            jd_text: The raw text of the job description.
            output_csv_path: The file path where the final submission CSV will be saved.
        """
        logger.info("Starting HirePulse pipeline execution...")
        
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

        # 3. Build Embeddings
        logger.info("Initializing Embedding Encoder...")
        try:
            encoder = EmbeddingEncoder()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedding encoder: {e}") from e
            
        logger.info("Encoding candidate texts and building FAISS index...")
        candidate_ids = [c.candidate_id for c in candidates]
        candidate_texts = [self._extract_candidate_text(c) for c in candidates]
        
        try:
            embeddings = encoder.encode_batch(candidate_texts)
            index = EmbeddingIndex(embedding_dim=encoder.embedding_dim)
            index.build(embeddings, candidate_ids)
        except Exception as e:
            raise RuntimeError(f"Failed to build embedding index: {e}") from e
        logger.info("Successfully built FAISS index.")

        # 4. Retrieval
        logger.info("Initializing retrieval strategies...")
        feature_strategy = FeatureBasedStrategy()
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
        ranked_results = ranking_engine.rank(retrieved_results)
        
        # Ensure strict secondary sorting (Score DESC, Candidate ID ASC)
        logger.info("Applying strict validation sorting...")
        sorted_results = sorted(
            ranked_results, 
            key=lambda r: (
                -(r.features.final_score if r.features.final_score is not None else 0.0), 
                r.candidate.candidate_id
            )
        )

        # 6. Explanations
        logger.info("Generating explanations...")
        explanation_engine = ExplanationEngine()
        
        export_items: List[Tuple[RetrievalResult, str]] = []
        for result in sorted_results:
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
