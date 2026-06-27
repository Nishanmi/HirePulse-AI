import csv
import json
import logging
import os
import sys

from backend.pipeline.runner import PipelineRunner

# Configure logging
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

def prepare_100_candidates(input_path: str, output_path: str) -> int:
    """
    Reads the 50-candidate sample, duplicates them to create exactly 100
    candidates with unique IDs, and saves them to a temporary file.
    This ensures we pass the pipeline's strict 100-candidate validation.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    original_count = len(data)
    if original_count >= 100:
        return original_count
        
    logger.info("Duplicating sample candidates to meet the 100-candidate requirement...")
    new_data = []
    
    # Add originals
    for i, cand in enumerate(data):
        new_data.append(cand)
        
    # Duplicate and generate unique IDs for the rest
    needed = 100 - len(new_data)
    for i in range(needed):
        # Deep copy using json serialization
        cand_copy = json.loads(json.dumps(data[i % original_count]))
        
        # Original format: CAND_XXXXXXX
        # We start unique IDs at CAND_9000000
        new_id = f"CAND_{9000000 + i}"
        cand_copy["candidate_id"] = new_id
        new_data.append(cand_copy)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2)
        
    return len(new_data)


def main() -> None:
    original_dataset = "data/raw/sample_candidates.json"
    temp_dataset = "data/raw/temp_100_candidates.json"
    output_dir = "output"
    output_csv = os.path.join(output_dir, "test_submission.csv")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Prepare 100 candidates
        total_loaded = prepare_100_candidates(original_dataset, temp_dataset)
        print("="*50)
        print(f"Total candidates prepared for pipeline: {total_loaded}")
        
        # Run Pipeline
        runner = PipelineRunner(top_k_retrieval=100)
        
        # PipelineRunner handles everything end-to-end
        runner.run(
            dataset_path=temp_dataset,
            jd_text=SAMPLE_JD_TEXT,
            output_csv_path=output_csv
        )
        
        # Verify CSV
        print("="*50)
        print("PIPELINE EXECUTION COMPLETE")
        print("="*50)
        print(f"Expected Output CSV Path: {output_csv}")
        
        if os.path.exists(output_csv):
            print(f"SUCCESS: CSV was created at {output_csv}")
            
            # Print first 5 rows
            print("-" * 50)
            print("FIRST 5 ROWS OF SUBMISSION CSV:")
            print("-" * 50)
            
            with open(output_csv, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                
                # Count total exported rows
                rows = list(reader)
                header = rows[0]
                data_rows = rows[1:]
                
                print(f"Total Ranked Candidates Exported: {len(data_rows)}")
                print(f"Header: {header}")
                
                for i, row in enumerate(data_rows[:5], 1):
                    # Truncate reasoning for clean printing
                    cand_id, rank, score, reasoning = row
                    short_reason = reasoning[:60] + "..." if len(reasoning) > 60 else reasoning
                    print(f"{i}. ID: {cand_id} | Rank: {rank} | Score: {score}")
                    print(f"   Reason: {short_reason}")
            print("-" * 50)
        else:
            print(f"ERROR: CSV was not created at {output_csv}")

    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        sys.exit(1)
        
    finally:
        # Cleanup temp file
        if os.path.exists(temp_dataset):
            os.remove(temp_dataset)


if __name__ == "__main__":
    main()
