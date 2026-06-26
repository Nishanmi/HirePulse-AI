import logging

from backend.candidate.parser import parse_candidates
from backend.jd.extractor import JDExtractor
from backend.features.engine import FeatureEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_JD = """
Title: Senior Backend Engineer
Company: Redrob
Location: Bangalore, India
Employment Type: Full-time

We are looking for a Senior Backend Engineer to build and scale our candidate
discovery and ranking platform. You will work on high-throughput data pipelines
and intelligent matching systems.

Requirements:
- Strong proficiency in Python
- Experience with REST APIs and microservices
- Solid understanding of SQL and NoSQL databases
- Familiarity with cloud platforms (AWS, GCP)

Nice to have:
- Experience with machine learning pipelines
- Knowledge of Elasticsearch or similar search engines
- Familiarity with Docker and Kubernetes

Domain:
- HR Tech
- SaaS

Responsibilities:
- Design and implement scalable backend services
- Optimize data pipelines for performance and reliability
- Collaborate with ML engineers on ranking algorithms
- Write clean, tested, and well-documented code

Behavioral:
- Strong problem-solving and analytical thinking
- Excellent communication skills
- Self-driven and proactive

Culture:
- Fast-paced startup environment
- Remote-friendly and flexible
- Continuous learning and innovation

Candidates should have 4 to 8 years of experience.
"""


def main():
    # Load candidates
    try:
        logger.info("Loading candidates from data/raw/sample_candidates.json...")
        candidates = parse_candidates("data/raw/sample_candidates.json")
        logger.info(f"Loaded {len(candidates)} candidates.")
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load candidates: {e}")
        return

    if not candidates:
        logger.error("No valid candidates found. Exiting.")
        return

    # Extract job description
    try:
        logger.info("Extracting job description...")
        extractor = JDExtractor()
        jd = extractor.extract(SAMPLE_JD)
        logger.info(f"Extracted JD for role: {jd.title}")
    except ValueError as e:
        logger.error(f"Failed to extract job description: {e}")
        return

    # Generate features for the first candidate
    try:
        first_candidate = candidates[0]
        logger.info(f"Generating features for candidate: {first_candidate.candidate_id}")

        engine = FeatureEngine()
        features = engine.extract_features(first_candidate, jd)

        print("\n--- Candidate Features ---")
        print(features.model_dump_json(indent=2))
        print("--------------------------\n")

        logger.info("Feature extraction complete.")
    except Exception as e:
        logger.exception(f"Failed to generate features: {e}")


if __name__ == "__main__":
    main()
