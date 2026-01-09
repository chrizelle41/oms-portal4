import os
import json
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI  # Use Azure client
from tqdm import tqdm

# --- PATH CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parents[1]
META_PATH = BASE_DIR / "outputs" / "documents_metadata.csv"
TEXT_PATH = BASE_DIR / "outputs" / "documents_text.jsonl"
OUTPUT_PATH = BASE_DIR / "outputs" / "documents_metadata_enriched.csv"

# --- LOAD ENVIRONMENT ---
load_dotenv()

# Initialize Azure OpenAI Client
endpoint = os.getenv("ENDPOINT_URL")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("OPENAI_API_VERSION", "2025-01-01-preview")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-5-chat")

client = None
if api_key and endpoint:
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )

def load_text_index():
    """Return dict: document_id -> text (shortened)."""
    text_map = {}
    if not TEXT_PATH.exists():
        return text_map

    with TEXT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            # Take a 4000 char sample for context
            text_map[rec["document_id"]] = rec["text"][:4000]  
    return text_map

def classify_doc(filename: str, building: str, text_sample: str) -> dict:
    """
    Call Azure AI to guess system, document_type, and asset_hint.
    """
    if client is None:
        return {"system": "Uncategorized", "document_type": "Document", "asset_hint": ""}

    prompt = f"""
You are helping organise building O&M (Operation and Maintenance) documentation.

Given the information below, classify the document into the correct category.

Building: {building}
Filename: {filename}

Short content sample from document:
{text_sample[:3000]}

Return a concise JSON object with exactly these fields:
- system: one of [HVAC, Electrical, Fire, Plumbing, Lifts, Security, General, Other]
- document_type: e.g. Manual, Warranty, Certificate, Commissioning Report, Test Report, Schedule, Spec, Drawing, Other
- asset_hint: short free text like "AHU-3", "Chiller", "Lighting", or "" if unclear.
"""

    try:
        resp = client.chat.completions.create(
            model=deployment,  # Must be your Azure Deployment Name
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        data = json.loads(resp.choices[0].message.content)
        return {
            "system": data.get("system", "Other"),
            "document_type": data.get("document_type", "Document"),
            "asset_hint": data.get("asset_hint", "")
        }
    except Exception as e:
        print(f"Error classifying {filename}: {e}")
        return {"system": "Error", "document_type": "Error", "asset_hint": ""}

def main():
    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not META_PATH.exists():
        raise SystemExit(f"Metadata CSV not found at: {META_PATH}. Run ingestion first.")

    print(f"Starting Metadata Enrichment using Azure Deployment: {deployment}")
    
    df = pd.read_csv(META_PATH)
    text_map = load_text_index()

    systems = []
    doc_types = []
    asset_hints = []

    # Process each row through Azure OpenAI
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Enriching metadata"):
        doc_id = row["document_id"]
        text_sample = text_map.get(doc_id, "")

        result = classify_doc(
            filename=row["filename"],
            building=row["building"],
            text_sample=text_sample
        )

        systems.append(result["system"])
        doc_types.append(result["document_type"])
        asset_hints.append(result["asset_hint"])

    # Update DataFrame with new AI-generated columns
    df["system"] = systems
    df["document_type"] = doc_types
    df["asset_hint"] = asset_hints

    # Save to the enriched CSV path
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSUCCESS: Saved enriched metadata to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()