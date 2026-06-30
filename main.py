import argparse
import logging
import os
import sys
from pathlib import Path

# Lock HuggingFace into offline mode permanently for hackathon safety
os.environ["HF_HUB_OFFLINE"] = "1"

from backend.pipeline.runner import PipelineRunner

def main():
    parser = argparse.ArgumentParser(description="HirePulse AI Candidate Ranking Pipeline")
    parser.add_argument(
        "--candidates", 
        required=True, 
        help="Path to the candidate dataset JSON file."
    )
    parser.add_argument(
        "--jd", 
        required=True, 
        help="Path to the job description text file."
    )
    parser.add_argument(
        "--out", 
        required=True, 
        help="Path to the output submission CSV file."
    )
    parser.add_argument(
        "--index-dir",
        required=False,
        help="Optional path to directory containing precomputed indexes (faiss.index, bm25.pkl, candidate_metadata.pkl)."
    )
    
    args = parser.parse_args()
    
    # Configure global logging for the CLI entry point
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    logger = logging.getLogger("hirepulse_cli")
    
    try:
        jd_path = Path(args.jd)
        if not jd_path.exists():
            logger.error("Job description file not found: %s", args.jd)
            sys.exit(1)
            
        logger.info("Reading Job Description from %s", args.jd)
        with open(jd_path, "r", encoding="utf-8") as f:
            jd_text = f.read()
            
        if not jd_text.strip():
            logger.error("Job description file is empty: %s", args.jd)
            sys.exit(1)
            
        logger.info("Initializing PipelineRunner...")
        runner = PipelineRunner(top_k_retrieval=10000)
        
        logger.info("Starting pipeline execution...")
        runner.run(
            dataset_path=args.candidates,
            jd_text=jd_text,
            output_csv_path=args.out,
            index_dir=args.index_dir
        )
        
        logger.info("Pipeline execution completed successfully.")
        sys.exit(0)
        
    except Exception as e:
        logger.error("Pipeline failed due to an error: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
