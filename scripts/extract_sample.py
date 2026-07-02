#!/usr/bin/env python3
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def extract_sample(input_path: str, output_path: str, num_samples: int = 100):
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        logger.error(f"Input file {input_path} not found.")
        sys.exit(1)
        
    logger.info(f"Extracting first {num_samples} records from {input_path}...")
    
    count = 0
    import gzip
    open_func = gzip.open if input_path.endswith('.gz') else open
    mode = 'rt' if input_path.endswith('.gz') else 'r'
    
    with open_func(input_file, mode, encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
         
        for line in f_in:
            if not line.strip():
                continue
                
            f_out.write(line)
            count += 1
            
            if count >= num_samples:
                break
                
    logger.info(f"Successfully wrote {count} records to {output_path}")

if __name__ == "__main__":
    extract_sample("data/raw/candidates.jsonl.gz", "data/raw/sample_100.jsonl", 100)
