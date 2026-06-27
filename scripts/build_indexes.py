#!/usr/bin/env python3
import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

# Ensure the project root is in sys.path so we can import 'backend'
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

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
        from backend.candidate.parser import stream_candidates_in_batches
        
        logger.info("Initializing Embedding Encoder...")
        encoder = EmbeddingEncoder()
        
        logger.info("Initializing FAISS index...")
        index = EmbeddingIndex(embedding_dim=encoder.embedding_dim)
        
        logger.info("Initializing BM25 preparation...")
        bm25_strategy = BM25Strategy()
        tokenised_corpus = []
        
        candidate_map = {}
        total_candidates = 0
        
        logger.info("Streaming candidates from %s...", args.candidates)
        
        for batch_idx, candidate_batch in enumerate(stream_candidates_in_batches(args.candidates, batch_size=5000)):
            if not candidate_batch:
                continue
                
            total_candidates += len(candidate_batch)
            logger.info("Processing batch %d (size: %d)...", batch_idx + 1, len(candidate_batch))
            
            # Extract texts and ids
            candidate_texts = [extract_candidate_text(c) for c in candidate_batch]
            candidate_ids = [c.candidate_id for c in candidate_batch]
            
            # Save to global map (stores Pydantic objects for metadata)
            for c in candidate_batch:
                candidate_map[c.candidate_id] = c
                
            # 1. FAISS embeddings
            logger.info("Generating embeddings for batch %d...", batch_idx + 1)
            # Use tuned batch_size for CPU L2/L3 cache
            embeddings = encoder.encode_batch(candidate_texts, batch_size=32)
            index.build(embeddings, candidate_ids)
            
            # 2. BM25 Accumulation
            logger.info("Tokenizing for BM25 (batch %d)...", batch_idx + 1)
            corpus = [bm25_strategy._build_candidate_document(c) for c in candidate_batch]
            batch_tokens = [bm25_strategy._tokenise(doc) for doc in corpus]
            tokenised_corpus.extend(batch_tokens)
            
        if total_candidates == 0:
            logger.error("No candidates found. Exiting.")
            sys.exit(1)
            
        logger.info("All batches processed. Total candidates: %d", total_candidates)

        # Finalize FAISS
        logger.info("Saving FAISS index...")
        faiss_path = output_dir / "faiss.index"
        faiss.write_index(index._index, str(faiss_path))
        logger.info("Saved FAISS index to %s", faiss_path)

        # Finalize BM25
        logger.info("Building final BM25 index (this may take a moment)...")
        bm25_index = BM25Okapi(tokenised_corpus)
        bm25_path = output_dir / "bm25.pkl"
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_index, f)
        logger.info("Saved BM25 index to %s", bm25_path)

        # Finalize Metadata
        logger.info("Saving candidate metadata...")
        metadata = {
            "candidates": candidate_map,
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
