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

# Fix pathing so 'src' is visible to FastAPI
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Importing vector search logic
from logic.vector_search import load_embeddings, search
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL PATHS ---
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

# --- FILE LISTING (FOR CHAT CONTEXT) ---
@app.get("/files")
def get_files():
    if not META_PATH.exists():
        return []
    
    file_lookup = {}
    if INPUT_ROOT.exists():
        for p in INPUT_ROOT.rglob("*"):
            if p.is_file() and not p.name.startswith('.'):
                rel_path = str(p.relative_to(INPUT_ROOT)).replace(os.sep, "/")
                file_lookup[p.name] = {
                    "path": rel_path,
                    "size": f"{round(p.stat().st_size / 1024, 1)} KB",
                    "date": pd.Timestamp(p.stat().st_mtime, unit='s').strftime("%Y-%m-%d")
                }

    df = pd.read_csv(META_PATH)
    records = df.fillna("").to_dict(orient="records")
    
    for rec in records:
        filename = rec.get("filename")
        compressed_name = f"{Path(filename).stem}_compressed{Path(filename).suffix}"
        match = file_lookup.get(filename) or file_lookup.get(compressed_name)
        
        if match:
            rec["document_id"] = match["path"]
            rec["size"] = match["size"]
            rec["date"] = match["date"]
        else:
            rec["size"] = "N/A"
            if not rec.get("document_id"):
                rec["document_id"] = filename

    return records

# --- CHAT & AI LOGIC ---
@app.post("/ask")
async def ask_ai(data: dict):
    query = (data.get("query") or "").strip()
    if not query:
        return {"error": "No query provided"}

    # --- 1) DECIDE MODE: AUDIT vs QA ---
    q = query.lower()
    audit_triggers = ["missing", "gap", "audit", "checklist", "status", "present", "do we have", "which documents", "availability"]
    qa_triggers = ["spec", "specification", "what does it say", "values", "dimensions", "density", "thermal", "summarise", "extract", "details"]
    
    is_audit = any(k in q for k in audit_triggers)
    is_qa = any(k in q for k in qa_triggers)
    
    mode = "qa" if is_qa else "audit"

    try:
        # --- 2) TARGETED FILENAME LOGIC (Your Working Logic) ---
        all_files_data = get_files() 
        priority_file = None
        for f in all_files_data:
            fname = f.get('filename', '').lower()
            if fname.replace(".pdf", "") in q or q in fname:
                priority_file = f
                break

        # --- 3) VECTOR SEARCH ---
        q_emb_resp = client.embeddings.create(
            model="text-embedding-3-large",
            input=query
        )
        q_emb = np.array(q_emb_resp.data[0].embedding, dtype="float32")
        
        docs = load_embeddings()  
        text_map = load_text_map()
        top = search(q_emb, docs, top_k=12) 

        # --- 4) BUILD CONTEXT ---
        context_blocks = []
        
        if priority_file:
            p_id = priority_file['document_id']
            context_blocks.append(
                f"PRIORITY_TARGET_FOUND: True\n"
                f"DISPLAY_NAME: {priority_file['filename']}\n"
                f"SYSTEM_PATH: {p_id}\n"
                f"TEXT: {text_map.get(p_id, '')[:3000]}"
            )

        for score, doc in top:
            doc_id = doc["document_id"]
            if priority_file and doc_id == priority_file['document_id']:
                continue 
                
            filename = doc_id.split('/')[-1] if '/' in doc_id else doc_id
            context_blocks.append(
                f"DISPLAY_NAME: {filename}\n"
                f"SYSTEM_PATH: {doc_id}\n"
                f"TEXT: {text_map.get(doc_id, '')[:2500]}"
            )
        
        full_context = "\n\n---\n\n".join(context_blocks)

        # --- 5) PROMPT LOGIC (DESIGN EDITS ONLY) ---
        if mode == "audit":
            system_msg = (
                "You are a professional O&M Auditor. Always start with 'Sure! I've analyzed the files...'\n"
                "STRICT AUDIT RULES:\n"
                "1. Output ONLY lines in format: Document Name | Status | Remark\n"
                "2. Status must be 'Present' or 'Missing'.\n"
                "3. Use exact DISPLAY_NAME from context for 'Present' items.\n"
                "4. If 'PRIORITY_TARGET_FOUND' is True and user asks for that file, only show that card.\n"
                "5. No headers."
            )
        else:
            system_msg = (
                "You are a helpful O&M Technical Assistant.\n"
                "The user is asking for SPECIFICATIONS or CONTENT.\n"
                "STRICT DESIGN RULES:\n"
                "1. Use **bold** for key terms and properties.\n"
                "2. Use standard Markdown bullet points (e.g., '- Property Name: Value').\n"
                "3. Structure your response with clear spacing so it is easy to read.\n"
                "4. AT THE END of your answer, if a document was used, provide the source EXACTLY as:\n"
                "SOURCE_FILE: [DISPLAY_NAME]\n"
                "5. Do NOT use the Table format (Name | Status | Remark) in this mode."
            )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Query: {query}\n\nContext:\n{full_context}"}
            ],
            temperature=0.1
        )

        return {
            "mode": mode,
            "answer": response.choices[0].message.content
        }

    except Exception as e:
        print(f"Error in ask_ai: {e}")
        return {"error": str(e)}
# --- PORTFOLIO & DOCUMENT MANAGEMENT ---
@app.get("/portfolio")
def get_portfolio_assets():
    if not INPUT_ROOT.exists():
        return {"stats": {"properties": 0, "docs": 0}, "assets": []}

    assets = []
    total_docs = 0
    folders = [f for f in INPUT_ROOT.iterdir() if f.is_dir() and not f.name.startswith('.')]
    
    for idx, folder in enumerate(sorted(folders)):
        files = [f for f in folder.rglob("*") if f.is_file() and f.name != "metadata.json" and not f.name.startswith('.')]
        file_count = len(files)
        total_docs += file_count
        
        name = folder.name.replace("_", " ")
        img = "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab"
        
        meta_file = folder / "metadata.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                    name = meta.get("name", name)
                    img = meta.get("image", img)
            except: pass

        assets.append({
            "id": f"asset-{idx}",
            "folder_name": folder.name, 
            "name": name,
            "docs": file_count,
            "status": "active",
            "img": img
        })

    return {"stats": {"properties": len(assets), "docs": total_docs}, "assets": assets}

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
            rel_path = str(file_path.relative_to(INPUT_ROOT)).replace(os.sep, "/")
            filename = file_path.name
            
            category, doc_type, asset_hint = "Uncategorized", "Document", ""
            
            if not db_df.empty:
                match = db_df[db_df['filename'] == filename]
                if not match.empty:
                    category = match.iloc[0].get('system') or "Uncategorized"
                    doc_type = match.iloc[0].get('document_type') or "Document"
                    asset_hint = match.iloc[0].get('asset_hint') or ""

            docs.append({
                "id": rel_path,
                "name": filename,
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
    if file_path.exists() and file_path.is_file():
        return FileResponse(path=file_path)

    path_obj = Path(document_id)
    compressed_name = f"{path_obj.stem}_compressed{path_obj.suffix}"
    
    # Fallback 1: Compressed in same folder
    compressed_path = INPUT_ROOT / path_obj.parent / compressed_name
    if compressed_path.exists():
        return FileResponse(path=compressed_path)

    # Fallback 2: Global filename search
    for p in INPUT_ROOT.rglob(path_obj.name):
        return FileResponse(path=p)
        
    for p in INPUT_ROOT.rglob(compressed_name):
        return FileResponse(path=p)

    return {"error": "File not found"}, 404

@app.post("/classify-document")
async def classify_document(file: UploadFile = File(...), folder_name: str = Form(...)):
    try:
        target_dir = INPUT_ROOT / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)
        save_path = target_dir / file.filename
        
        with save_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "Return ONLY: HVAC, Electrical, Fire, or Plumbing."},
                      {"role": "user", "content": f"Classify: {file.filename}"}],
            temperature=0
        )
        category = response.choices[0].message.content.strip()

        return {
            "document_id": f"{folder_name}/{file.filename}",
            "filename": file.filename,
            "system": category,
            "document_type": "Uploaded Document",
            "size": f"{round(save_path.stat().st_size / 1024, 1)} KB",
            "date": pd.Timestamp.now().strftime("%Y-%m-%d")
        }
    except Exception as e:
        return {"error": str(e)}, 500

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
        meta_file = target_dir / "metadata.json"
        with open(meta_file, "w") as f:
            json.dump({"name": asset_name, "location": location, "image": image_url}, f)
        return {"status": "success", "folder_name": folder_name}
    except Exception as e:
        return {"error": str(e)}, 500

@app.delete("/portfolio/{folder_name}/docs/{filename}")
async def delete_document(folder_name: str, filename: str):
    file_path = INPUT_ROOT / folder_name / filename
    try:
        if file_path.exists():
            file_path.unlink()
            return {"status": "success", "message": f"{filename} deleted"}
        return {"error": "File not found"}, 404
    except Exception as e:
        return {"error": str(e)}, 500

@app.delete("/portfolio/assets/{folder_name}")
async def delete_asset_folder(folder_name: str):
    folder_path = INPUT_ROOT / folder_name
    try:
        if folder_path.exists() and folder_path.is_dir():
            shutil.rmtree(folder_path)
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
        metadata = {}
        if meta_file.exists():
            with open(meta_file, "r") as f:
                metadata = json.load(f)
        if "name" in data: metadata["name"] = data["name"]
        if "image" in data: metadata["image"] = data["image"]
        with open(meta_file, "w") as f:
            json.dump(metadata, f)
        return {"status": "success", "metadata": metadata}
    except Exception as e:
        return {"error": str(e)}, 500