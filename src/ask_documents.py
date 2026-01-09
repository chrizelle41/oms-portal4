import os
import json
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI  # Updated import

from src.vector_search import load_embeddings, search

# --- PATH CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parents[1]
META_PATH = BASE_DIR / "outputs" / "documents_metadata_enriched.csv"
TEXT_PATH = BASE_DIR / "outputs" / "documents_text.jsonl"

# --- LOAD ENVIRONMENT ---
load_dotenv()

# Initialize Azure OpenAI Client
client = AzureOpenAI(
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION", "2025-01-01-preview")
)

# Deployment names from .env
chat_deployment = os.getenv("DEPLOYMENT_NAME", "gpt-5-chat")
embedding_deployment = os.getenv("EMBEDDING_DEPLOYMENT", "text-embedding-3-large")

def load_text_map():
    text_map = {}
    if not TEXT_PATH.exists():
        print(f"[ERROR] Text index not found at {TEXT_PATH}")
        return text_map

    with TEXT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text_map[rec["document_id"]] = rec["text"]
    return text_map

def main():
    query = input("\nAsk your documents: ")
    if not query.strip():
        return

    print("Searching for relevant information...")

    try:
        # 1. Embed query using Azure Deployment
        q_emb_resp = client.embeddings.create(
            model=embedding_deployment,
            input=query
        )
        q_emb = np.array(q_emb_resp.data[0].embedding, dtype="float32")

        # 2. Vector Search
        docs = load_embeddings()
        text_map = load_text_map()
        
        # We don't necessarily need meta here for the LLM context, 
        # but it's available if you want to pull building names, etc.
        # meta = pd.read_csv(META_PATH) 

        top = search(q_emb, docs, top_k=5)

        context_blocks = []
        for score, doc in top:
            doc_id = doc["document_id"]
            # Get text and cap it to prevent context window overflow
            text = text_map.get(doc_id, "No text found.")[:2500]
            context_blocks.append(f"SOURCE: {doc_id} (Relevance: {score:.3f})\nCONTENT: {text}\n")

        full_context = "\n\n".join(context_blocks)

        # 3. AI Generation using Azure Deployment
        response = client.chat.completions.create(
            model=chat_deployment,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a professional O&M document assistant. Use the provided context to answer the user's query accurately. If the answer isn't in the context, say you don't know based on the current files."
                },
                {
                    "role": "user", 
                    "content": f"Context from documents:\n{full_context}\n\nUser Question: {query}"
                }
            ],
            temperature=0.1
        )

        print("\n--- AI AUDITOR RESPONSE ---")
        print(response.choices[0].message.content)
        print("---------------------------\n")

    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")

if __name__ == "__main__":
    main()