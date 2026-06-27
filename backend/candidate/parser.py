import json
import logging
from pathlib import Path
from typing import List, Union

from pydantic import ValidationError

from backend.models import Candidate

logger = logging.getLogger(__name__)

def parse_candidates(file_path: Union[str, Path]) -> List[Candidate]:
    """
    Load and parse a JSON file containing candidate records.

    Args:
        file_path (Union[str, Path]): Path to the JSON file containing candidate records.

    Returns:
        List[Candidate]: A list of successfully validated Candidate models.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file contains malformed JSON data or is not a list.
    """
    path = Path(file_path)
    
    if not path.is_file():
        raise FileNotFoundError(f"Candidate data file not found: {path}")
        
    try:
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix.lower() == '.jsonl':
                data = []
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            else:
                data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in candidate data file {path}: {e}") from e

    # Assume data is a list of candidate dictionaries
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list or JSONL lines of candidate records, got {type(data).__name__}")
        
    candidates = []
    
    for idx, record in enumerate(data):
        try:
            candidate = Candidate.model_validate(record)
            candidates.append(candidate)
        except ValidationError as e:
            candidate_id = record.get("candidate_id", f"unknown_at_index_{idx}") if isinstance(record, dict) else f"invalid_type_at_index_{idx}"
            logger.warning(f"Skipping invalid candidate record ({candidate_id}): {e}")
            
    logger.info(f"Successfully parsed {len(candidates)} out of {len(data)} candidate records from {path}")
    
    return candidates

def stream_candidates_in_batches(file_path: Union[str, Path], batch_size: int = 5000):
    """
    Load and parse a JSONL file containing candidate records in chunks.
    Yields batches of Candidate models to prevent massive memory spikes.
    """
    path = Path(file_path)
    
    if not path.is_file():
        raise FileNotFoundError(f"Candidate data file not found: {path}")
        
    if path.suffix.lower() != '.jsonl':
        # Fallback for standard JSON: unfortunately we have to load it all, but we can yield in chunks
        logger.info(f"Streaming from standard JSON file (loads into memory first): {path}")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            batch = []
            for record in data:
                try:
                    candidate = Candidate.model_validate(record)
                    batch.append(candidate)
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
                except ValidationError as e:
                    logger.warning(f"Skipping invalid candidate record: {e}")
            if batch:
                yield batch
        return

    # Efficient streaming for JSONL
    logger.info(f"Efficiently streaming from JSONL file: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        batch = []
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
                
            try:
                record = json.loads(line)
                candidate = Candidate.model_validate(record)
                batch.append(candidate)
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"Skipping invalid record at line {idx}: {e}")
                continue
                
            if len(batch) >= batch_size:
                yield batch
                batch = []
                
        if batch:
            yield batch
