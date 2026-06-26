import logging
from typing import List

from backend.candidate.parser import parse_candidates
from backend.embeddings.encoder import EmbeddingEncoder
from backend.embeddings.index import EmbeddingIndex
from backend.models import JobDescription
from backend.retrieval.embedding import EmbeddingStrategy

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Create 5 realistic candidate profile texts
CANDIDATE_TEXTS = [
    # Candidate 1
    (
        "Senior Backend Engineer with 6 years of experience specializing in Python, FastAPI, Django, "
        "and PostgreSQL. Highly skilled in designing RESTful APIs, optimizing database queries, "
        "and deploying microservices on AWS using Docker. Passionate about building scalable architectures."
    ),
    # Candidate 2
    (
        "React Frontend Developer with 4 years of experience building responsive web applications. "
        "Expertise in JavaScript, TypeScript, CSS grid/flexbox, and HTML5. Experienced in state management "
        "using Redux and integrating REST APIs. Focuses on premium UI/UX design and smooth micro-animations."
    ),
    # Candidate 3
    (
        "Machine Learning Engineer and Data Scientist with 5 years of experience. Proficient in PyTorch, "
        "scikit-learn, and SQL. Developed NLP and transformer-based models for document classification. "
        "Experienced with data analysis, feature engineering, and deploying models to CPU/GPU instances."
    ),
    # Candidate 4
    (
        "DevOps and Cloud Infrastructure Engineer. Over 5 years of experience in CI/CD pipeline automation, "
        "infrastructure as code (IaC) using Terraform, container orchestration with Kubernetes and Docker, "
        "and system monitoring. Proven track record of improving system uptime and reliability."
    ),
    # Candidate 5
    (
        "Backend Python Developer with 3 years of experience. Skilled in writing clean, PEP 8 compliant code. "
        "Experience with Flask, SQL databases, pandas, and web scraping. Familiar with version control (Git) "
        "and setting up simple automated test suites."
    ),
]

# Create a realistic sample Job Description
SAMPLE_JD = JobDescription(
    title="Senior Python Backend Developer",
    must_have_skills=["Python", "FastAPI", "Django", "PostgreSQL"],
    responsibilities=[
        "Design and implement scalable backend services",
        "Optimize database queries",
        "Deploy microservices on AWS using Docker",
    ],
    keywords=["FastAPI", "Django", "PostgreSQL", "AWS", "Docker"],
)


def main() -> None:
    try:
        # Load 5 candidates from sample candidate data
        logger.info("Loading sample candidates...")
        candidates = parse_candidates("data/raw/sample_candidates.json")[:5]
        if len(candidates) < 5:
            raise ValueError(f"Expected at least 5 candidates, found {len(candidates)}")

        candidate_ids = [c.candidate_id for c in candidates]
        logger.info(f"Loaded candidates: {candidate_ids}")

        # Initialize EmbeddingEncoder
        logger.info("Initializing EmbeddingEncoder...")
        encoder = EmbeddingEncoder()

        # Encode candidates
        logger.info("Encoding candidate profile texts...")
        embeddings = encoder.encode_batch(CANDIDATE_TEXTS)

        # Build FAISS index
        logger.info("Building FAISS index...")
        index = EmbeddingIndex(embedding_dim=encoder.embedding_dim)
        index.build(embeddings, candidate_ids)

        # Initialize EmbeddingStrategy
        logger.info("Initializing EmbeddingStrategy...")
        strategy = EmbeddingStrategy(encoder=encoder, index=index)

        # Score the candidates against the Job Description
        logger.info("Scoring candidates using EmbeddingStrategy...")
        scores = strategy.score_candidates(candidates, SAMPLE_JD)

        # Sort candidates based on scores descending
        sorted_results = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        print("\n--- Embedding Strategy Retrieval Results (Top 3) ---")
        for candidate_id, score in sorted_results[:3]:
            print(f"candidate_id: {candidate_id}")
            print(f"similarity score: {score:.6f}")
            print("-" * 52)
        print()

        logger.info("Embedding strategy test completed successfully!")

    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
    except ValueError as e:
        logger.error(f"Value error: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
