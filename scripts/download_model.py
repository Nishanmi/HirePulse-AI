#!/usr/bin/env python3
import os
import argparse
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser(description="Download Hugging Face models for offline use.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="BAAI/bge-small-en-v1.5", 
        help="The Hugging Face model ID to download."
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data/models/bge-small-en-v1.5", 
        help="The local directory to save the model."
    )
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Downloading model '{args.model}' to '{args.output_dir}'...")
    print("This may take a few minutes depending on your internet connection.")
    
    # Download the model files
    snapshot_download(
        repo_id=args.model,
        local_dir=args.output_dir,
        local_dir_use_symlinks=False, # We want the actual files, not symlinks to the global cache
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "onnx/*"], # Ignore unused large formats if present
    )
    
    print("Download complete! The model is now ready for offline use.")
    print(f"Make sure backend/embeddings/encoder.py points to '{args.output_dir}'.")

if __name__ == "__main__":
    main()
