import os
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI  # Updated import
from tqdm import tqdm

# --- PATH CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parents[1]
TEXT_PATH = BASE_DIR / "outputs" / "documents_text.jsonl"
EMBED_PATH = BASE_DIR / "outputs" / "document_embeddings.jsonl"

# --- LOAD ENVIRONMENT ---
load_dotenv()

# Initialize Azure OpenAI Client
client = AzureOpenAI(
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION", "2025-01-01-preview")
)

# Deployment name for your embedding model (e.g., "text-embedding-3-large")
embedding_deployment = os.getenv("EMBEDDING_DEPLOYMENT", "text-embedding-3-large")

def main():
    if not TEXT_PATH.exists():
        raise SystemExit(f"Text file not found: {TEXT_PATH}. Please run ingest_documents.py first.")

    print(f"Loading text data from {TEXT_PATH}...")
    with TEXT_PATH.open("r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f]

    print(f"Generating embeddings using Azure Deployment: {embedding_deployment}")
    
    with EMBED_PATH.open("w", encoding="utf-8") as out:
        for rec in tqdm(lines, desc="Embedding documents"):
            # Azure standard models usually have an 8192 token limit
            # We cap at 8000 characters as a safe approximation for text
            text = rec["text"][:8000]  
            doc_id = rec["document_id"]

            try:
                # Call Azure OpenAI Embeddings API
                emb = client.embeddings.create(
                    model=embedding_deployment, 
                    input=text
                )

                # Write record to JSONL
                out.write(json.dumps({
                    "document_id": doc_id,
                    "embedding": emb.data[0].embedding
                }) + "\n")
                
            except Exception as e:
                print(f"\n[WARN] Failed to embed document {doc_id}: {e}")
                continue

    print(f"\nSUCCESS: Saved all embeddings to {EMBED_PATH}")

if __name__ == "__main__":
    main()