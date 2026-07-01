#!/usr/bin/env python3
import argparse
import logging
import os
import pickle
import sys

# Lock HuggingFace into offline mode permanently for hackathon safety
os.environ["HF_HUB_OFFLINE"] = "1"
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
from backend.features.engine import FeatureEngine

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
        
        # Open metadata file in truncate/append mode at the start
        metadata_path = output_dir / "candidate_metadata.jsonl"
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write("") # Truncate if exists
            
        total_candidates = 0
        
        logger.info("Streaming candidates from %s...", args.candidates)
        
        for batch_idx, candidate_batch in enumerate(stream_candidates_in_batches(args.candidates, batch_size=1000)):
            if not candidate_batch:
                continue
                
            total_candidates += len(candidate_batch)
            logger.info("Processing batch %d (size: %d)...", batch_idx + 1, len(candidate_batch))
            
            candidate_ids = []
            candidate_texts = []
            
            for c in candidate_batch:
                candidate_ids.append(c.candidate_id)
                candidate_texts.append(extract_candidate_text(c))
                
            # Compute Role Embeddings
            logger.info("Generating role embeddings for batch %d...", batch_idx + 1)
            role_titles = []
            for c in candidate_batch:
                c.normalized_role_title = FeatureEngine.normalize_role_title(c.profile.current_title)
                # If empty, just use a dummy string to avoid encoder errors, or keep it empty
                role_titles.append(c.normalized_role_title if c.normalized_role_title else "unknown")
                
            role_embeddings = encoder.encode_batch(role_titles, batch_size=32, max_length=16)
            for c, emb in zip(candidate_batch, role_embeddings):
                if c.normalized_role_title:
                    c.role_embedding = emb.tolist()

            # Compute Career Evidence Embeddings
            logger.info("Generating career evidence embeddings for batch %d...", batch_idx + 1)
            career_texts = []
            for c in candidate_batch:
                titles = [FeatureEngine.normalize_role_title(c.profile.current_title)]
                for role in c.career_history[:3]:
                    cleaned = FeatureEngine.normalize_role_title(role.title)
                    if cleaned:
                        titles.append(cleaned)
                career_text = " | ".join(t for t in titles if t)
                c.career_role_text = career_text if career_text else None
                career_texts.append(career_text if career_text else "unknown")

            career_embeddings = encoder.encode_batch(career_texts, batch_size=32, max_length=128)
            for c, emb in zip(candidate_batch, career_embeddings):
                if c.career_role_text:
                    c.career_role_embedding = emb.tolist()

            # Compute Career Description Embeddings (what candidates actually built)
            logger.info("Generating career description embeddings for batch %d...", batch_idx + 1)
            desc_texts = []
            for c in candidate_batch:
                descs = [role.description for role in c.career_history if role.description]
                desc_text = " ".join(descs).strip()
                desc_texts.append(desc_text if desc_text else "unknown")

            desc_embeddings = encoder.encode_batch(desc_texts, batch_size=32, max_length=256)
            for c, desc_text, emb in zip(candidate_batch, desc_texts, desc_embeddings):
                if desc_text != "unknown":
                    c.career_desc_embedding = emb.tolist()
                
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
            
            # 3. Stream Metadata to disk and instantly free RAM
            with open(metadata_path, "a", encoding="utf-8") as f:
                for c in candidate_batch:
                    f.write(c.model_dump_json() + "\n")
                    
            # 4. Checkpointing
            if (batch_idx + 1) % 5 == 0:
                logger.info("Saving checkpoint (FAISS and BM25) at batch %d...", batch_idx + 1)
                faiss.write_index(index._index, str(output_dir / "faiss.index"))
                with open(output_dir / "bm25.pkl", "wb") as f:
                    pickle.dump(BM25Okapi(tokenised_corpus), f)
                with open(output_dir / "faiss_map.pkl", "wb") as f:
                    pickle.dump(index._candidate_map, f)
            
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

        # Finalize Metadata Map (Only FAISS IDs mapping, very small)
        logger.info("Saving FAISS ID map...")
        faiss_map_path = output_dir / "faiss_map.pkl"
        with open(faiss_map_path, "wb") as f:
            pickle.dump(index._candidate_map, f)
        logger.info("Saved FAISS map to %s", faiss_map_path)
        
        logger.info("All indexes built successfully! Metadata streamed to %s", metadata_path)
        
        logger.info("All indexes built successfully!")
        
    except Exception as e:
        logger.error("Failed to build indexes: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
