import sys
import os
import json
import shutil
import urllib.parse
from pathlib import Path

import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from fastapi.staticfiles import StaticFiles


# Fix pathing so 'src' is visible to FastAPI
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# --- GLOBAL PATHS ---
# --- GLOBAL PATHS (Updated for Flat Structure) ---
INPUT_ROOT = current_dir / "Input_Documents" 
OUTPUT_DIR = current_dir / "outputs"
META_PATH = OUTPUT_DIR / "documents_metadata_enriched.csv"
TEXT_PATH = OUTPUT_DIR / "documents_text.jsonl"
ENRICHED_META = OUTPUT_DIR / "documents_metadata_enriched.csv"
INPUT_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# REVERTED: Importing from 'logic' folder
from logic.vector_search import load_embeddings, search
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://oms-portal-uifinal.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Use the specific list instead of just "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"] # This helps the browser see the response
)

# --- ADD THIS AFTER CORS MIDDLEWARE ---
app.mount("/direct_preview", StaticFiles(directory=str(INPUT_ROOT)), name="direct_preview")

# --- MODELS ---
class LoginRequest(BaseModel):
    email: str
    password: str

    # --- LOGIN ENDPOINT ---
# --- 1. FIXED LOGIN ENDPOINT ---
@app.post("/login")
async def login(data: LoginRequest):
    # This pulls from Render's 'Environment' variables
    SECRET_PASSWORD = os.getenv("DEMO_PASSWORD", "Virtual2026!")
    
    email_lower = data.email.lower()
    if not email_lower.endswith("@virtualviewing.com"):
        raise HTTPException(
            status_code=403, 
            detail="Access restricted to @virtualviewing.com users."
        )

    if data.password != SECRET_PASSWORD:
        raise HTTPException(
            status_code=401, 
            detail="Invalid password."
        )

    return {
        "status": "success",
        "user": {"email": data.email, "role": "admin"},
        "token": "demo-session-token"
    }

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
    if not META_PATH.exists():
        return []
    
    df = pd.read_csv(META_PATH)
    records = df.fillna("").to_dict(orient="records")
    
    for rec in records:
        # Use document_id to find the file on disk
        file_id = rec.get("document_id")
        if file_id:
            file_path = INPUT_ROOT / file_id
            if file_path.exists():
                # FRONTEND EXPECTS: 'filename', 'size', 'system'
                rec["filename"] = file_id.split('/')[-1] # Ensure filename is present
                rec["size"] = f"{round(file_path.stat().st_size / 1024, 1)} KB"
                rec["date"] = pd.Timestamp(file_path.stat().st_mtime, unit='s').strftime("%Y-%m-%d")
            else:
                rec["size"] = "N/A"
    return records

@app.post("/ask")
async def ask_ai(data: dict):
    query = data.get("query")
    if not query: 
        return {"error": "No query provided"}

    REQUIRED_CHECKLIST = [
        "HVAC Operation & Maintenance Manuals", 
        "Electrical Schematic Layouts", 
        "Commissioning & Test Reports", 
        "BMS Software Specifications", 
        "F-Gas Compliance Certificates",
        "Fire Safety Certificates",
        "Asset Register"
    ]

    try:
        # --- 1. INTENT DETECTION ---
        q_lower = query.lower()
        audit_triggers = ["missing", "gap", "audit", "checklist", "what is not here", "availability"]
        is_audit_mode = any(word in q_lower for word in audit_triggers)

        # --- 2. Targeted Logic for "Present" files ---
        all_files_data = get_files() 
        priority_file = None
        for f in all_files_data:
            fname = f['filename'].lower()
            if fname.replace(".pdf", "") in q_lower or q_lower in fname:
                priority_file = f
                break

        # --- 3. OpenAI Embedding Search ---
        q_emb_resp = client.embeddings.create(
            model="text-embedding-3-large", 
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
                f"TEXT: {text_map.get(priority_file['document_id'], '')[:1500]}"
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

        # --- 4. AI Generation (Dual Mode) ---
        if is_audit_mode:
            # MODE A: Audit / Missing Cards
            system_msg = (
                "You are a professional O&M Auditor. The user is specifically asking for a gap analysis or missing files.\n"
                f"REQUIRED_CHECKLIST: {', '.join(REQUIRED_CHECKLIST)}\n\n"
                "STRICT RULES:\n"
                "1. Compare REQUIRED_CHECKLIST against the Context. If an item is not found, list it as 'Missing'.\n"
                "2. FORMAT: Document Name | Status | Remark\n"
                "3. Status must be 'Present' or 'Missing'.\n"
                # FIXED: Instruction to use SYSTEM_PATH for the link to work
                "4. For present items, the 'Remark' MUST be the exact SYSTEM_PATH from the context. No headers."
            )
        else:
            # MODE B: Normal Q&A
            system_msg = (
                "You are a helpful O&M Assistant. Answer the user's question based ONLY on the provided context.\n"
                "1. If the user asks about specifications, values, or details, provide a clear text answer with bold terms.\n"
                # FIXED: Instructing AI to use SYSTEM_PATH so the preview doesn't load endlessly
                "2. If you use information from a specific file, you MUST end your response with: SOURCE_FILE: [SYSTEM_PATH]\n"
                "3. If the user mentions a specific filename, use the PRIORITY_TARGET text first.\n"
                "4. Do NOT use the 'Name | Status | Remark' format unless specifically asked for an audit."
            )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_msg},
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
            
        # Standard OpenAI Classification
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "Return ONLY: HVAC, Electrical, Fire, or Plumbing."},
                      {"role": "user", "content": f"Classify: {file.filename}"}],
            temperature=0
        )
        category = response.choices[0].message.content.strip()

        # IMPORTANT: Return the FULL object so the frontend can show it immediately
        return {
        "id": f"{folder_name}/{file.filename}", # This fixed the preview!
        "filename": file.filename,                       # This fixed the blank name!
        "cat": category,
        "document_type": "Uploaded Document",
        "size": f"{round(save_path.stat().st_size / 1024, 1)} KB",
        "date": pd.Timestamp.now().strftime("%Y-%m-%d")
        }
    except Exception as e:
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
    for file_path in target_folder.rglob("*"):
        if file_path.is_file() and not file_path.name.startswith('.') and file_path.name != "metadata.json":
            # This 'rel_path' must match the document_id in your system
            rel_path = str(file_path.relative_to(INPUT_ROOT)).replace(os.sep, "/")
            filename = file_path.name
            
            # Match metadata from CSV
            category = "Uncategorized"
            doc_type = "Document"
            if not db_df.empty:
                match = db_df[db_df['document_id'] == rel_path]
                if not match.empty:
                    category = match.iloc[0].get('system') or "Uncategorized"
                    doc_type = match.iloc[0].get('document_type') or "Document"

            docs.append({
                "document_id": rel_path, # App.jsx uses this for preview
                "id": rel_path,
                "filename": filename,    # App.jsx uses this for titles
                "name": filename,
                "system": category,      # App.jsx uses 'system' for categories
                "cat": category,
                "doc_type": doc_type,
                "status": "Verified" if category != "Uncategorized" else "Processing",
                "date": pd.Timestamp(file_path.stat().st_mtime, unit='s').strftime("%Y-%m-%d"),
                "size": f"{round(file_path.stat().st_size / 1024, 1)} KB",
            })
    return docs

@app.get("/preview/{document_id:path}")
async def preview_document(document_id: str):
    decoded_id = urllib.parse.unquote(document_id)
    file_path = (INPUT_ROOT / decoded_id).resolve()
    
    if not file_path.exists():
        print(f"ERROR: File not found at {file_path}")
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(
        path=file_path,
        media_type='application/pdf',
        headers={
            "Content-Disposition": "inline", # Forces browser to view inside iframe
            "X-Frame-Options": "ALLOWALL",    # Allows iframe loading
            "Access-Control-Allow-Origin": "*"
        }
    )

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
