import csv
import logging
from typing import List, Tuple

from backend.retrieval.retriever import RetrievalResult

logger = logging.getLogger(__name__)

class SubmissionExporter:
    """
    Exports ranked candidates to the standard CSV format required by the
    Hackathon's validate_submission.py script.
    """
    
    def export(self, ranked_items: List[Tuple[RetrievalResult, str]], output_path: str) -> None:
        """
        Exports the ranked candidates with their explanations to a CSV file.
        
        Args:
            ranked_items: A list of tuples containing (RetrievalResult, explanation_string).
                          The list MUST be sorted by:
                          1. final_score (descending)
                          2. candidate_id (ascending for equal scores)
            output_path: Path to the output CSV file.
            
        Raises:
            ValueError: If there are not exactly 100 candidates, or if any data is invalid.
        """
        if len(ranked_items) != 100:
            raise ValueError(f"Submission requires exactly 100 ranked candidates. Got {len(ranked_items)}.")
            
        # 1. TIE-BREAKING: Ensure strictly deterministic sorting before export
        # Sort by final_score descending, then candidate_id ascending for ties
        ranked_items.sort(key=lambda x: (-x[0].features.final_score, x[0].candidate.candidate_id))
            
        # Basic validation pass before writing
        for rank, (result, reasoning) in enumerate(ranked_items, start=1):
            if not result.candidate or not getattr(result.candidate, "candidate_id", None):
                raise ValueError(f"Missing candidate or candidate_id at rank {rank}")
            score = result.features.final_score
            if score is None:
                raise ValueError(f"Candidate {result.candidate.candidate_id} at rank {rank} is missing a final_score")
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Candidate {result.candidate.candidate_id} has invalid final_score {score}. "
                    "Expected a normalized score between 0 and 1."
                )
            if not isinstance(reasoning, str) or not reasoning.strip():
                raise ValueError(
                    f"Candidate {result.candidate.candidate_id} at rank {rank} has an empty reasoning"
                )
                
        try:
            with open(output_path, mode='w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["candidate_id", "rank", "score", "reasoning"])
                
                for rank, (result, reasoning) in enumerate(ranked_items, start=1):
                    writer.writerow([
                        result.candidate.candidate_id,
                        rank,
                        f"{result.features.final_score:.6f}",
                        reasoning.strip()
                    ])
            logger.info("Successfully exported %d candidates to %s", len(ranked_items), output_path)
            
        except IOError as e:
            logger.error("Failed to write to %s: %s", output_path, e)
            raise
