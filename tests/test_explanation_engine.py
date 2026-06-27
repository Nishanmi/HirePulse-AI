import logging

from backend.candidate.parser import parse_candidates
from backend.jd.extractor import JDExtractor
from backend.retrieval.retriever import FeatureBasedStrategy
from backend.retrieval.bm25 import BM25Strategy
from backend.embeddings.encoder import EmbeddingEncoder
from backend.embeddings.index import EmbeddingIndex
from backend.retrieval.embedding import EmbeddingStrategy
from backend.retrieval.hybrid import HybridRetriever
from backend.ranking.engine import RankingEngine
from backend.explainability.explanation_engine import ExplanationEngine

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_JD_TEXT = """
Title: Senior AI / Backend Engineer
Company: Redrob
Location: Remote
Employment Type: Full-time

We are looking for an experienced Senior AI Engineer to join our core intelligence team at Redrob.
You will be building state-of-the-art candidate matching and ranking systems.

Requirements:
- Strong proficiency in Python and modern data science stacks
- Solid understanding of ML algorithms and NLP techniques
- Experience with PyTorch or TensorFlow

Nice to have:
- Experience building recommendation systems
- Familiarity with Pydantic and FastAPI

Domain:
- HR Tech
- Candidate Discovery

Responsibilities:
- Design and implement ranking algorithms
- Build robust and scalable ML pipelines
- Collaborate with cross-functional teams to improve matching accuracy

Behavioral:
- Strong problem-solving skills
- Excellent communication and ability to explain complex ML concepts
- Team player

Culture:
- Fast-paced and innovative environment
- Remote-first and flexible
- Continuous learning and growth

You should have 5 to 8 years of experience.
"""

def _extract_candidate_text(candidate) -> str:
    """Helper to extract text from a candidate for embedding."""
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

def main() -> None:
    try:
        # Load candidates
        logger.info("Loading candidates from data/raw/sample_candidates.json...")
        candidates = parse_candidates("data/raw/sample_candidates.json")
        logger.info(f"Loaded {len(candidates)} candidates.")

        if not candidates:
            logger.error("No valid candidates found. Exiting.")
            return

        # Parse JD
        logger.info("Extracting job description...")
        extractor = JDExtractor()
        jd = extractor.extract(SAMPLE_JD_TEXT)
        logger.info(f"Extracted JD for role: {jd.title}")

        # Prepare Embeddings
        logger.info("Initializing EmbeddingEncoder...")
        encoder = EmbeddingEncoder()
        
        logger.info("Encoding candidate profile texts...")
        candidate_ids = [c.candidate_id for c in candidates]
        candidate_texts = [_extract_candidate_text(c) for c in candidates]
        
        embeddings = encoder.encode_batch(candidate_texts)
        
        logger.info("Building FAISS index...")
        index = EmbeddingIndex(embedding_dim=encoder.embedding_dim)
        index.build(embeddings, candidate_ids)

        # Initialize strategies
        logger.info("Initializing retrieval strategies...")
        feature_strategy = FeatureBasedStrategy()
        bm25_strategy = BM25Strategy()
        embedding_strategy = EmbeddingStrategy(encoder=encoder, index=index, top_k=100)

        # Initialize HybridRetriever
        logger.info("Initializing HybridRetriever...")
        hybrid_retriever = HybridRetriever(
            strategies=[
                (feature_strategy, 1.0),
                (bm25_strategy, 1.5),
                (embedding_strategy, 2.0),
            ],
            top_k=5,
            relevance_threshold=0.01
        )

        # Run retrieval
        logger.info("Running HybridRetriever to fetch top 5 candidates...")
        retrieval_results = hybrid_retriever.retrieve(candidates, jd)
        
        if not retrieval_results:
            logger.warning("No candidates retrieved. Exiting.")
            return

        # Initialize RankingEngine
        logger.info("Initializing RankingEngine...")
        ranking_engine = RankingEngine()
        
        # Rank candidates
        logger.info("Ranking candidates...")
        ranked_results = ranking_engine.rank(retrieval_results)

        # Initialize ExplanationEngine
        logger.info("Initializing ExplanationEngine...")
        explanation_engine = ExplanationEngine()

        # Print results
        print("\n" + "="*40)
        print("EXPLANATION ENGINE TEST RESULTS")
        print("="*40)
        
        for rank, result in enumerate(ranked_results, start=1):
            explanation = explanation_engine.explain(result.candidate, result.features, jd)
            print("-" * 40)
            print(f"Rank: {rank}")
            print(f"Candidate ID: {result.candidate.candidate_id}")
            print(f"Final Score: {result.features.final_score:.6f}")
            print(f"Explanation: {explanation}")
            
        print("-" * 40)
        print()
        logger.info("Explanation engine test completed successfully!")

    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
    except ValueError as e:
        logger.error(f"Value error: {e}")
    except AssertionError as e:
        logger.error(f"Assertion error: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
