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
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in candidate data file {path}: {e}") from e

    # Assume data is a list of candidate dictionaries
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of candidate records, got {type(data).__name__}")
        
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
