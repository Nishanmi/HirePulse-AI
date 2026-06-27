import logging
from backend.jd.extractor import JDExtractor as JobDescriptionExtractor

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

SAMPLE_JD = """
Title: Senior AI Engineer
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

def main():
    extractor = JobDescriptionExtractor()
    
    try:
        logger.info("Parsing sample Job Description...")
        job_description = extractor.extract(SAMPLE_JD)
        
        print("\n--- Parsed Job Description ---")
        # Use Pydantic v2's model_dump_json for a clean, formatted output
        print(job_description.model_dump_json(indent=2))
        print("------------------------------\n")
        
        logger.info("Successfully parsed the job description.")
        
    except ValueError as e:
        logger.error(f"Failed to parse job description: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during parsing: {e}")

if __name__ == "__main__":
    main()
