#!/usr/bin/env python3
import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import faiss
from rank_bm25 import BM25Okapi

from backend.candidate.parser import parse_candidates
from backend.embeddings.encoder import EmbeddingEncoder
from backend.embeddings.index import EmbeddingIndex
from backend.retrieval.bm25 import BM25Strategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def extract_candidate_text(candidate) -> str:
    """Helper to extract and format candidate text for embeddings."""
    parts = [
        candidate.profile.headline,
        candidate.profile.summary,
        candidate.profile.current_title,
        candidate.profile.current_company,
        candidate.profile.current_industry,
    ]
    for role in candidate.career_history:
        parts.extend([role.title, role.company, role.industry, role.description])
    for edu in candidate.education:
        parts.extend([edu.institution, edu.degree, edu.field_of_study])
    for skill in candidate.skills:
        parts.append(skill.name)
        
    return " ".join(filter(None, parts))

def main():
    parser = argparse.ArgumentParser(description="Pre-compute retrieval artifacts for HirePulse.")
    parser.add_argument("--candidates", required=True, help="Path to candidate JSON file.")
    args = parser.parse_args()

    output_dir = Path("indexes")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Parse candidates
        logger.info("Parsing candidates from %s...", args.candidates)
        candidates = parse_candidates(args.candidates)
        logger.info("Successfully parsed %d candidates.", len(candidates))

        if not candidates:
            logger.error("No candidates found. Exiting.")
            sys.exit(1)

        # 2. Generate embeddings
        logger.info("Initializing Embedding Encoder...")
        encoder = EmbeddingEncoder()
        
        logger.info("Extracting candidate text...")
        candidate_texts = [extract_candidate_text(c) for c in candidates]
        candidate_ids = [c.candidate_id for c in candidates]
        
        logger.info("Generating candidate embeddings (this may take a while)...")
        embeddings = encoder.encode_batch(candidate_texts)
        logger.info("Embeddings generated.")

        # 3. Build FAISS index
        logger.info("Building FAISS index...")
        index = EmbeddingIndex(embedding_dim=encoder.embedding_dim)
        index.build(embeddings, candidate_ids)
        
        faiss_path = output_dir / "faiss.index"
        faiss.write_index(index._index, str(faiss_path))
        logger.info("Saved FAISS index to %s", faiss_path)

        # 4. Build BM25 index
        logger.info("Building BM25 index...")
        bm25_strategy = BM25Strategy()
        corpus = [bm25_strategy._build_candidate_document(c) for c in candidates]
        tokenised_corpus = [bm25_strategy._tokenise(doc) for doc in corpus]
        bm25_index = BM25Okapi(tokenised_corpus)
        
        bm25_path = output_dir / "bm25.pkl"
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_index, f)
        logger.info("Saved BM25 index to %s", bm25_path)

        # 5. Save candidate metadata
        logger.info("Saving candidate metadata...")
        # Store both the candidate objects and the FAISS mapping
        metadata = {
            "candidates": {c.candidate_id: c for c in candidates},
            "faiss_map": index._candidate_map
        }
        meta_path = output_dir / "candidate_metadata.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)
        logger.info("Saved candidate metadata to %s", meta_path)
        
        logger.info("All indexes built successfully!")
        
    except Exception as e:
        logger.error("Failed to build indexes: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
