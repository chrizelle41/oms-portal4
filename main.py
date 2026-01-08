import sys
import os
import json
import shutil
from pathlib import Path

import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
from openai import AzureOpenAI

# Fix pathing so 'src' is visible to FastAPI
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import from src
from logic.vector_search import load_embeddings, search

load_dotenv()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION", "2025-01-01-preview")
)

deployment_name = os.getenv("DEPLOYMENT_NAME", "gpt-5-chat")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "https://oms-portal-uifinal.onrender.com" # Add your actual Render frontend URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL PATHS ---
# --- GLOBAL PATHS (Updated for Flat Structure) ---
INPUT_ROOT = current_dir / "Input_Documents" 
OUTPUT_DIR = current_dir / "outputs"
META_PATH = OUTPUT_DIR / "documents_metadata_enriched.csv"
TEXT_PATH = OUTPUT_DIR / "documents_text.jsonl"
ENRICHED_META = OUTPUT_DIR / "documents_metadata_enriched.csv"

def load_text_map():
    text_map = {}
    if not TEXT_PATH.exists():
        return text_map
    with TEXT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text_map[rec["document_id"]] = rec["text"]
    return text_map

@app.get("/files")
def get_files():
    """
    Returns all files merged with their physical disk metadata (like size)
    to ensure the Preview Drawer always has accurate information.
    """
    if not META_PATH.exists():
        return []
    
    df = pd.read_csv(META_PATH)
    records = df.fillna("").to_dict(orient="records")
    
    # Enrich records with physical file size if missing
    for rec in records:
        file_id = rec.get("document_id")
        if file_id:
            file_path = INPUT_ROOT / file_id
            if file_path.exists():
                # Ensure 'size' is populated for AllFilesPage and Chat Preview
                rec["size"] = f"{round(file_path.stat().st_size / 1024, 1)} KB"
                # Ensure 'date' is consistent
                rec["date"] = pd.Timestamp(file_path.stat().st_mtime, unit='s').strftime("%Y-%m-%d")
            else:
                rec["size"] = "N/A"
    
    return records

@app.post("/ask")
async def ask_ai(data: dict):
    query = data.get("query")
    if not query: 
        return {"error": "No query provided"}

    try:
        # 1. Targeted Logic for "Present" files
        all_files_data = get_files() 
        priority_file = None
        clean_query = query.lower()
        
        for f in all_files_data:
            fname = f['filename'].lower()
            if fname.replace(".pdf", "") in clean_query or clean_query in fname:
                priority_file = f
                break

        # 2. Azure-Specific Embedding Search
        # Note: Ensure you have a deployment for your embedding model in Azure
        q_emb_resp = client.embeddings.create(
            model=os.getenv("EMBEDDING_DEPLOYMENT", "text-embedding-3-large"),
            input=query
        )
        q_emb = np.array(q_emb_resp.data[0].embedding, dtype="float32")
        
        docs = load_embeddings()  
        text_map = load_text_map()
        top = search(q_emb, docs, top_k=15) 

        context_blocks = []
        
        if priority_file:
            context_blocks.append(
                f"PRIORITY_TARGET_FOUND: True\n"
                f"DISPLAY_NAME: {priority_file['filename']}\n"
                f"SYSTEM_PATH: {priority_file['document_id']}\n"
                f"TEXT: {text_map.get(priority_file['document_id'], '')[:1000]}"
            )

        for score, doc in top:
            doc_id = doc["document_id"]
            if priority_file and doc_id == priority_file['document_id']:
                continue
                
            filename = doc_id.split('/')[-1] if '/' in doc_id else doc_id
            context_blocks.append(
                f"DISPLAY_NAME: {filename}\n"
                f"SYSTEM_PATH: {doc_id}\n"
                f"TEXT: {text_map.get(doc_id, '')[:3000]}"
            )
        
        full_context = "\n\n".join(context_blocks)

        # 3. Azure AI Generation 
        # model= here MUST be your deployment name (gpt-5-chat)
        response = client.chat.completions.create(
            model=os.getenv("DEPLOYMENT_NAME", "gpt-5-chat"),
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are a professional O&M Auditor. Always start with a friendly intro like 'Sure! I've analyzed the files, and here is the status...'"
                        "\n\nSTRICT AUDIT RULES:\n"
                        "1. TARGETED MODE: If 'PRIORITY_TARGET_FOUND' is True and the user is asking for that specific file, ONLY provide the card for that one file.\n"
                        "2. GAP ANALYSIS: Look for AC components (Manuals, Layouts, Commissioning, BMS, F-Gas). If missing, list as 'Missing'.\n"
                        "3. COMPREHENSIVE: List EVERY missing item. Do not summarize.\n"
                        "4. FOR PRESENT DOCUMENTS: Use exact 'DISPLAY_NAME' as the Document Name.\n"
                        "5. FILTERING: Only show 'Missing' if asked for missing, 'Present' if asked for existing.\n"
                        "6. FORMAT: Document Name | Status | Remark\n"
                        "7. NO HEADERS: Do not include 'Document Name | Status'."
                    )
                },
                {"role": "user", "content": f"Query: {query}\n\nContext:\n{full_context}"}
            ],
            temperature=0.1
        )
        return {"answer": response.choices[0].message.content}

    except Exception as e:
        print(f"Error in ask_ai: {e}")
        return {"error": str(e)}

@app.get("/portfolio")
def get_portfolio_assets():
    # If the folder doesn't exist, return empty stats
    if not INPUT_ROOT.exists():
        print(f"DEBUG: INPUT_ROOT not found at {INPUT_ROOT}")
        return {"stats": {"properties": 0, "docs": 0}, "assets": []}

    assets = []
    total_docs = 0
    
    # Get all subdirectories (buildings)
    folders = [f for f in INPUT_ROOT.iterdir() if f.is_dir() and not f.name.startswith('.')]
    
    for idx, folder in enumerate(sorted(folders)):
        # Count all files inside this building folder (recursively)
        files = [f for f in folder.rglob("*") if f.is_file() and f.name != "metadata.json" and not f.name.startswith('.')]
        file_count = len(files)
        total_docs += file_count
        
        # Default name and image
        name = folder.name.replace("_", " ")
        img = "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab"
        
        # Look for custom name/image in metadata.json
        meta_file = folder / "metadata.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                    name = meta.get("name", name)
                    img = meta.get("image", img)
            except: 
                pass

        assets.append({
            "id": f"asset-{idx}",
            "folder_name": folder.name, 
            "name": name,
            "docs": file_count,
            "status": "active",
            "img": img
        })

    return {
        "stats": {
            "properties": len(assets),
            "docs": total_docs
        },
        "assets": assets
    }

@app.get("/portfolio/{folder_name}/docs")
def get_folder_docs(folder_name: str):
    target_folder = INPUT_ROOT / folder_name
    if not target_folder.exists():
        return []

    # Load the enriched database to look up metadata
    db_df = pd.DataFrame()
    if ENRICHED_META.exists():
        db_df = pd.read_csv(ENRICHED_META).replace({np.nan: None}) #

    docs = []
    for file_path in target_folder.rglob("*"):
        # Filter out metadata.json and hidden files
        if file_path.is_file() and not file_path.name.startswith('.') and file_path.name != "metadata.json":
            rel_path = str(file_path.relative_to(INPUT_ROOT)).replace(os.sep, "/")
            filename = file_path.name
            
            # Default values if no match is found
            category = "Uncategorized"
            doc_type = "Document"
            asset_hint = "None"
            
            # Match the file on disk to the metadata in the CSV
            if not db_df.empty:
                match = db_df[db_df['filename'] == filename]
                if not match.empty:
                    category = match.iloc[0].get('system') or "Uncategorized"
                    doc_type = match.iloc[0].get('document_type') or "Document"
                    asset_hint = match.iloc[0].get('asset_hint') or "None"

            docs.append({
                "id": rel_path,
                "name": filename,
                "cat": category, # Mapped to 'system'
                "doc_type": doc_type, # Mapped to 'document_type'
                "asset_hint": asset_hint,
                "date": pd.Timestamp(file_path.stat().st_mtime, unit='s').strftime("%Y-%m-%d"),
                "size": f"{round(file_path.stat().st_size / 1024, 1)} KB",
                "user": "System"
            })
    return docs
@app.post("/classify-document")
async def classify_document(
    file: UploadFile = File(...), 
    folder_name: str = Form(...) 
):
    try:
        target_dir = INPUT_ROOT / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = target_dir / file.filename
        
        with save_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        prompt = f"Classify this O&M document filename into exactly one category: HVAC, Electrical, Fire, or Plumbing. Filename: {file.filename}"
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "Return ONLY the category name: HVAC, Electrical, Fire, or Plumbing."},
                      {"role": "user", "content": prompt}],
            temperature=0
        )
        category = response.choices[0].message.content.strip().replace(".", "")

        return {
            "document_id": f"{folder_name}/{file.filename}",
            "category": category,
            "name": file.filename,
            "status": "Verified"
        }
    except Exception as e:
        print(f"Classification Error: {e}")
        return {"error": str(e)}, 500
    
@app.get("/portfolio/{folder_name}/docs")
def get_folder_docs(folder_name: str):
    target_folder = INPUT_ROOT / folder_name
    if not target_folder.exists():
        return []

    db_df = pd.DataFrame()
    if ENRICHED_META.exists():
        db_df = pd.read_csv(ENRICHED_META).replace({np.nan: None})

    docs = []
    for idx, file_path in enumerate(target_folder.rglob("*")):
        if file_path.is_file() and not file_path.name.startswith('.'):
            rel_path = str(file_path.relative_to(INPUT_ROOT)).replace(os.sep, "/")
            filename = file_path.name
            
            category = "Uncategorized"
            doc_type = "Document"
            asset_hint = ""
            
            if not db_df.empty:
                match = db_df[db_df['filename'] == filename]
                if not match.empty:
                    category = match.iloc[0].get('system') or "Uncategorized"
                    doc_type = match.iloc[0].get('document_type') or "Document"
                    asset_hint = match.iloc[0].get('asset_hint') or ""

            docs.append({
                "id": rel_path,
                "name": filename,
                "lang": "EN",
                "cat": str(category),
                "doc_type": str(doc_type),
                "asset_hint": str(asset_hint),
                "status": "Verified" if category != "Uncategorized" else "Processing",
                "date": pd.Timestamp(file_path.stat().st_mtime, unit='s').strftime("%Y-%m-%d"),
                "size": f"{round(file_path.stat().st_size / 1024, 1)} KB",
                "user": "System"
            })
    return docs

@app.get("/preview/{document_id:path}")
async def preview_document(document_id: str):
    file_path = INPUT_ROOT / document_id
    if not file_path.exists():
        return {"error": f"File not found at {file_path}"}, 404
    return FileResponse(path=file_path)

@app.post("/create-asset")
async def create_asset(data: dict):
    asset_name = data.get("name")
    location = data.get("location", "Unknown Location")
    image_url = data.get("image", "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab")
    
    if not asset_name:
        return {"error": "Name is required"}, 400
    
    folder_name = asset_name.replace(" ", "_")
    target_dir = INPUT_ROOT / folder_name
    
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata so it persists after refresh
        meta_file = target_dir / "metadata.json"
        with open(meta_file, "w") as f:
            json.dump({
                "name": asset_name,
                "location": location,
                "image": image_url
            }, f)
            
        return {"status": "success", "folder_name": folder_name}
    except Exception as e:
        return {"error": str(e)}, 500
    

    # Endpoint to delete a specific file inside an asset folder
@app.delete("/portfolio/{folder_name}/docs/{filename}")
async def delete_document(folder_name: str, filename: str):
    file_path = INPUT_ROOT / folder_name / filename
    try:
        if file_path.exists():
            file_path.unlink()  # Deletes the file
            return {"status": "success", "message": f"{filename} deleted"}
        return {"error": "File not found"}, 404
    except Exception as e:
        return {"error": str(e)}, 500

# Endpoint to delete an entire asset folder
@app.delete("/portfolio/assets/{folder_name}")
async def delete_asset_folder(folder_name: str):
    folder_path = INPUT_ROOT / folder_name
    try:
        if folder_path.exists() and folder_path.is_dir():
            shutil.rmtree(folder_path)  # Deletes folder and all contents
            return {"status": "success", "message": f"Asset {folder_name} deleted"}
        return {"error": "Folder not found"}, 404
    except Exception as e:
        return {"error": str(e)}, 500
    
@app.patch("/portfolio/assets/{folder_name}")
async def update_asset_metadata(folder_name: str, data: dict):
    folder_path = INPUT_ROOT / folder_name
    meta_file = folder_path / "metadata.json"
    
    if not folder_path.exists():
        return {"error": "Asset not found"}, 404

    try:
        # Load existing or create new
        metadata = {}
        if meta_file.exists():
            with open(meta_file, "r") as f:
                metadata = json.load(f)
        
        # Update fields
        if "name" in data:
            metadata["name"] = data["name"]
        if "image" in data:
            metadata["image"] = data["image"]
            
        with open(meta_file, "w") as f:
            json.dump(metadata, f)
            
        return {"status": "success", "metadata": metadata}
    except Exception as e:
        return {"error": str(e)}, 500