import logging
import tempfile
from pathlib import Path
import numpy as np

from backend.embeddings.encoder import EmbeddingEncoder
from backend.embeddings.index import EmbeddingIndex

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Create 5 realistic candidate profile texts
CANDIDATE_PROFILES = [
    # Candidate 1: Backend Developer
    {
        "id": "cand_001",
        "text": (
            "Senior Backend Engineer with 6 years of experience specializing in Python, FastAPI, Django, "
            "and PostgreSQL. Highly skilled in designing RESTful APIs, optimizing database queries, "
            "and deploying microservices on AWS using Docker. Passionate about building scalable architectures."
        ),
    },
    # Candidate 2: React Developer
    {
        "id": "cand_002",
        "text": (
            "React Frontend Developer with 4 years of experience building responsive web applications. "
            "Expertise in JavaScript, TypeScript, CSS grid/flexbox, and HTML5. Experienced in state management "
            "using Redux and integrating REST APIs. Focuses on premium UI/UX design and smooth micro-animations."
        ),
    },
    # Candidate 3: Data Scientist / ML Engineer
    {
        "id": "cand_003",
        "text": (
            "Machine Learning Engineer and Data Scientist with 5 years of experience. Proficient in PyTorch, "
            "scikit-learn, and SQL. Developed NLP and transformer-based models for document classification. "
            "Experienced with data analysis, feature engineering, and deploying models to CPU/GPU instances."
        ),
    },
    # Candidate 4: DevOps / Infrastructure Engineer
    {
        "id": "cand_004",
        "text": (
            "DevOps and Cloud Infrastructure Engineer. Over 5 years of experience in CI/CD pipeline automation, "
            "infrastructure as code (IaC) using Terraform, container orchestration with Kubernetes and Docker, "
            "and system monitoring. Proven track record of improving system uptime and reliability."
        ),
    },
    # Candidate 5: Python Backend / Data Engineer
    {
        "id": "cand_005",
        "text": (
            "Backend Python Developer with 3 years of experience. Skilled in writing clean, PEP 8 compliant code. "
            "Experience with Flask, SQL databases, pandas, and web scraping. Familiar with version control (Git) "
            "and setting up simple automated test suites."
        ),
    },
]

# Create a sample job description
SAMPLE_JD = (
    "Looking for a Senior Python Developer with strong backend experience. "
    "Must be proficient in Django or FastAPI, PostgreSQL databases, and AWS cloud services. "
    "Experience with containerization using Docker and microservices architecture is highly preferred."
)


def main() -> None:
    try:
        # Initialize encoder
        logger.info("Initializing EmbeddingEncoder...")
        encoder = EmbeddingEncoder()
        logger.info(f"Loaded encoder model. Dimension: {encoder.embedding_dim}")

        # Encode candidates
        logger.info("Encoding candidate profile texts...")
        candidate_ids = [c["id"] for c in CANDIDATE_PROFILES]
        candidate_texts = [c["text"] for c in CANDIDATE_PROFILES]

        # Use batch encoding for efficiency
        embeddings = encoder.encode_batch(candidate_texts)
        logger.info(f"Generated {len(embeddings)} candidate embeddings of shape {embeddings.shape}.")

        # Build FAISS index
        logger.info("Building FAISS index...")
        index = EmbeddingIndex(embedding_dim=encoder.embedding_dim)
        index.build(embeddings, candidate_ids)
        logger.info("FAISS index built successfully.")

        # Encode job description
        logger.info("Encoding sample job description...")
        query_embedding = encoder.encode_job_description(SAMPLE_JD)
        logger.info(f"Generated query embedding of shape {query_embedding.shape}.")

        # Search the index for the top 3 nearest candidates
        logger.info("Searching index for top 3 nearest candidates...")
        results = index.search(query_embedding, top_k=3)

        print("\n--- Search Results ---")
        for candidate_id, score in results:
            print(f"candidate_id: {candidate_id}")
            print(f"similarity score: {score:.6f}")
            print("-" * 22)
        print()

        # Test save() and load() functionality
        logger.info("Testing save() and load() functionality...")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger.info(f"Saving index to temporary path: {temp_path}")
            index.save(temp_path)

            logger.info("Loading index back from saved path...")
            new_index = EmbeddingIndex(embedding_dim=encoder.embedding_dim)
            new_index.load(temp_path)

            # Search with loaded index and verify identical results
            logger.info("Verifying search results on loaded index...")
            loaded_results = new_index.search(query_embedding, top_k=3)

            assert len(results) == len(loaded_results), "Result counts differ"
            for (orig_id, orig_score), (load_id, load_score) in zip(results, loaded_results):
                assert orig_id == load_id, f"Candidate IDs do not match: {orig_id} vs {load_id}"
                assert np.isclose(orig_score, load_score, atol=1e-5), (
                    f"Scores do not match for {orig_id}: {orig_score} vs {load_score}"
                )

            logger.info("FAISS index save and load functionality verified successfully!")

    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
    except ValueError as e:
        logger.error(f"Value error: {e}")
    except AssertionError as e:
        logger.error(f"Assertion failed during index validation: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
