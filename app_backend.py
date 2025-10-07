import io
import os
import sqlite3
import uuid
import fitz
import pandas as pd
import numpy as np
import pickle
import json
import random
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, List, Dict, Literal, Optional, Union
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi import Depends, FastAPI, Header, UploadFile, File, BackgroundTasks, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai
import time
from PIL import Image
from docx import Document as DocxDocument
import requests
from auth import router as auth_router, get_current_user
import warnings
warnings.filterwarnings("ignore", message="Unsupported Windows version")

load_dotenv()

#import nltk

# Add path to your offline nltk_data folder
#nltk.data.path.append(r"nltk_data")  

# ----------------------------
# Directories & DB path
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "app_data.db")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
INTER_DIR = os.path.join(BASE_DIR, "data", "intermediate")
STORE_ROOT = os.path.join(BASE_DIR, "stores")
FEEDBACK_DIR = os.path.join(BASE_DIR, "data", "feedback")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(INTER_DIR, exist_ok=True)
os.makedirs(STORE_ROOT, exist_ok=True)
os.makedirs(FEEDBACK_DIR, exist_ok=True)

# Path to feedback storage file
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "feedback.json")

conversation_memory: Dict[str, List[Dict]] = {}
MAX_MEMORY_LENGTH = 20  # keep last 20 messages

# ----------------------------
# SQLite utilities
# ----------------------------
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)

conn = get_connection()
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,  -- UUID
    username TEXT UNIQUE,
    password_hash TEXT,
    role TEXT DEFAULT 'user',  -- 'admin' or 'user'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# Optional: Queries table to track all queries
c.execute("""
CREATE TABLE IF NOT EXISTS queries (
    id TEXT PRIMARY KEY,
    conversation_id TEXT,
    project_id INTEGER,
    query TEXT,
    answer TEXT,
    user_id TEXT,
    response_time FLOAT,
    attempts INTEGER DEFAULT 1,
    is_fallback BOOLEAN DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    precision1 REAL,
    precision3 REAL,
    precision5 REAL,
    claim_support REAL,
    query_coverage REAL,
    num_claims INTEGER
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT
)
""")


c.execute("""CREATE TABLE IF NOT EXISTS user_project (
    user_id TEXT,
    project_id INTEGER,
    role TEXT DEFAULT 'user',
    PRIMARY KEY(user_id, project_id)
)"""
)

c.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER,
    name TEXT,
    path TEXT,
    type TEXT,
    status TEXT DEFAULT 'pending',
    progress INTEGER DEFAULT 0,
    uploaded_by TEXT
)
""")
c.execute("""
CREATE TABLE IF NOT EXISTS doc_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER,
    chunk_type TEXT,
    content TEXT,
    metadata TEXT
)
""")
c.execute("""
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    query_id TEXT,
    project_id INTEGER,
    query TEXT,
    answer TEXT,
    contexts TEXT,
    feedback TEXT,
    comment TEXT,
    user_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()
conn.close()

def create_indexes():
    conn = get_connection()
    c = conn.cursor()
    c.execute("CREATE INDEX IF NOT EXISTS idx_queries_conv_proj_user ON queries(conversation_id, project_id, user_id)")
    conn.commit()
    conn.close()
    
# ----------------------------
# Embedding model
# ----------------------------
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ----------------------------
# FAISS per-project cache
# ----------------------------
class ProjectIndex:
    def __init__(self, vectorstore: FAISS):
        self.vectorstore = vectorstore

project_cache: Dict[int, ProjectIndex] = {}

def project_dir(project_id: int) -> str:
    d = os.path.join(STORE_ROOT, str(project_id))
    os.makedirs(d, exist_ok=True)
    return d

def load_project_index(project_id: int) -> ProjectIndex:
    if project_id in project_cache:
        return project_cache[project_id]

    d = project_dir(project_id)
    vs_path = os.path.join(d, "faiss_store.pkl")
    vectorstore = None
    if os.path.exists(vs_path):
        try:
            vectorstore = pickle.load(open(vs_path, "rb"))
        except Exception as e:
            print(f"Failed to load persisted vectorstore: {e}")
    pi = ProjectIndex(vectorstore=vectorstore)
    project_cache[project_id] = pi
    return pi

def persist_project_index(project_id: int):
    pi = project_cache.get(project_id)
    if not pi or not pi.vectorstore:
        return
    d = project_dir(project_id)
    try:
        pickle.dump(pi.vectorstore, open(os.path.join(d, "faiss_store.pkl"), "wb"))
    except Exception as e:
        print(f"Error persisting vectorstore {project_id}: {e}")

# ----------------------------
# Gemini (LLM) setup
# ----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai_client = genai.Client(api_key=GEMINI_API_KEY)

def redact_pii(text: str) -> str:
    """Simple regex-based PII redaction (email, phone, ID, etc)."""
    if not text.strip():
        return text

    # Common regex patterns for PII
    patterns = {
        "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "PHONE": r"\+?\d[\d\s\-]{8,}\d",
        "ID": r"\b[A-Z]{2,3}\d{4,}\b",  # e.g., PAN/employee IDs
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    }

    redacted_text = text
    for label, pattern in patterns.items():
        redacted_text = re.sub(pattern, f"[{label}_REDACTED]", redacted_text)

    return redacted_text


def sanitize_user_query(query: str) -> str:

    # 1. Basic prompt injection check
    forbidden_phrases = [
        "ignore previous instructions",
        "forget your instructions",
        "bypass safety",
        "malicious",
        "--",  # comment injection
        "drop table",    # dangerous SQL
        "delete from",   # dangerous SQL
        "update ",   # dangerous SQL
    ]
    for phrase in forbidden_phrases:
        query = re.sub(re.escape(phrase), "[REDACTED]", query, flags=re.IGNORECASE)

    # 2. Strip leading/trailing spaces
    return query.strip()


from unstructured.partition.auto import partition

# ----------------------------
# Load & split with Unstructured
# ----------------------------
def split_textual_document(filepath: str, max_chunk_size: int = 1000) -> List[Document]:
    """
    Improved chunking with smart grouping and relational metadata:
    - Groups related elements (table descriptions + tables)
    - Adds metadata linking for relational queries
    - Maintains clean, focused chunks
    - Preserves hierarchy and context
    """
    elements = partition(filename=filepath)
    documents = []

    hierarchy_stack = []  # Track active heading levels
    current_group = []  # Group related elements together
    chunk_counter = 0

    def flush_group():
        """Flush current group as one or more documents with relational metadata."""
        nonlocal current_group, chunk_counter
        if not current_group:
            return

        # Group 1: Table descriptions with their tables
        grouped_elements = []
        temp_group = []
        
        for el in current_group:
            if el.category == "Table" and temp_group and temp_group[-1].category in ["UncategorizedText", "NarrativeText"]:
                # Table follows a description - group them
                temp_group.append(el)
            else:
                if temp_group:
                    grouped_elements.append(temp_group)
                temp_group = [el]
        if temp_group:
            grouped_elements.append(temp_group)

        # Create documents for each group
        for group in grouped_elements:
            if len(group) == 1:
                # Single element - create individual chunk
                el = group[0]
                text = el.text.strip()
                if not text:
                    continue
                    
                # Split large elements but maintain context
                if len(text) > max_chunk_size:
                    chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
                    for idx, chunk in enumerate(chunks):
                        documents.append(create_document(
                            chunk, el, hierarchy_stack, filepath, chunk_counter
                        ))
                        chunk_counter += 1
                else:
                    documents.append(create_document(
                        text, el, hierarchy_stack, filepath, chunk_counter
                    ))
                    chunk_counter += 1
            else:
                # Multiple related elements - combine if they fit
                combined_text = "\n".join([el.text.strip() for el in group if el.text.strip()])
                if len(combined_text) <= max_chunk_size:
                    # Combine into one chunk
                    primary_el = group[0]
                    documents.append(create_document(
                        combined_text, primary_el, hierarchy_stack, filepath, chunk_counter,
                        related_elements=len(group),
                        element_types=[el.category for el in group]
                    ))
                    chunk_counter += 1
                else:
                    # Keep separate but add linking metadata
                    for i, el in enumerate(group):
                        text = el.text.strip()
                        if not text:
                            continue
                            
                        doc = create_document(
                            text, el, hierarchy_stack, filepath, chunk_counter,
                            group_id=chunk_counter // len(group),  # Same group ID for related chunks
                            element_index=i,
                            total_elements=len(group)
                        )
                        documents.append(doc)
                        chunk_counter += 1

        current_group = []

    def create_document(content, element, hierarchy, filepath, chunk_id, **extra_metadata):
        """Helper to create Document with consistent metadata."""
        metadata = {
            "category": element.category,
            "filename": Path(filepath).name,
            "hierarchy": hierarchy.copy(),
            "parent_heading": hierarchy[-1] if hierarchy else None,
            "chunk_id": chunk_id,
            "element_id": id(element),
        }
        metadata.update(extra_metadata)
        redacted_content = redact_pii(content)
        return Document(page_content=redacted_content, metadata=metadata)

    # Process elements
    for el in elements:
        if not el.text.strip():
            continue

        if el.category == "Title":
            # Flush current group before new heading
            flush_group()
            if hierarchy_stack:
                hierarchy_stack.pop()
            hierarchy_stack.append(el.text.strip())
            
            # Title gets its own chunk
            current_group = [el]
            flush_group()
        else:
            current_group.append(el)

    # Flush final group
    flush_group()

    # Add sequential linking metadata for navigation
    add_sequential_links(documents)
    
    return documents

def add_sequential_links(documents):
    """Add metadata to link sequential chunks for better relational retrieval."""
    for i in range(len(documents)):
        if i > 0:
            # Link to previous chunk
            prev_id = documents[i-1].metadata.get("chunk_id")
            if prev_id is not None:
                documents[i].metadata["prev_chunk_id"] = prev_id
        if i < len(documents) - 1:
            # Link to next chunk
            next_id = documents[i+1].metadata.get("chunk_id")
            if next_id is not None:
                documents[i].metadata["next_chunk_id"] = next_id
        
        # Add section context for relational queries
        if "group_id" in documents[i].metadata:
            documents[i].metadata["section_type"] = "related_group"


def split_tabular_document(filepath: str, batch_size: int = 50):
    """Parse CSV/XLSX into schema + row batches for embedding."""
    ext = Path(filepath).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(filepath)
        sheets = {"Sheet1": df}
    else:  # Excel
        xls = pd.ExcelFile(filepath)
        sheets = {name: xls.parse(name) for name in xls.sheet_names}

    documents = []
    for sheet, df in sheets.items():
        # Schema / header info
        documents.append(Document(
            page_content=f"Sheet: {sheet}\nColumns: {', '.join(df.columns)}",
            metadata={"category": "schema", "sheet": sheet}
        ))

        # Row batches
        for start in range(0, len(df), batch_size):
            batch = df.iloc[start:start + batch_size]
            batch_text = f"Sheet: {sheet}\nRows {start}-{start+len(batch)-1}:\n{batch.to_csv(index=False)}"
            
            redacted_content = redact_pii(batch_text)
            
            documents.append(Document(
                page_content=redacted_content,
                metadata={"category": "rows", "sheet": sheet, "batch": f"{start}-{start+len(batch)-1}"}
            ))
    return documents

def split_pdf_document(filepath: str, max_chunk_size: int = 1000) -> List[Document]:
    """
    Extract text from PDF using PyMuPDF and split into Document chunks
    using same logic as split_textual_document.
    """

    # Step 1: Extract text from PDF
    elements = []
    try:
        with fitz.open(filepath) as pdf_doc:
            for page_num, page in enumerate(pdf_doc):
                text = page.get_text().strip()
                if text:
                    # Treat each page as an element
                    elements.append(type("Element", (), {"text": text, "category": "Page"}))
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {e}")

    # Step 2: Split like textual document
    documents = []
    hierarchy_stack = []
    current_group = []
    chunk_counter = 0

    def flush_group():
        nonlocal current_group, chunk_counter
        if not current_group:
            return
        grouped_elements = [current_group]  # simple grouping for PDF pages
        for group in grouped_elements:
            combined_text = "\n".join([el.text.strip() for el in group if el.text.strip()])
            if len(combined_text) <= max_chunk_size:
                documents.append(create_document(combined_text, group[0], hierarchy_stack, filepath, chunk_counter))
                chunk_counter += 1
            else:
                # Split large chunks
                text = combined_text
                chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
                for chunk in chunks:
                    documents.append(create_document(chunk, group[0], hierarchy_stack, filepath, chunk_counter))
                    chunk_counter += 1
        current_group.clear()

    def create_document(content, element, hierarchy, filepath, chunk_id, **extra_metadata):
        metadata = {
            "category": element.category,
            "filename": Path(filepath).name,
            "hierarchy": hierarchy.copy(),
            "parent_heading": hierarchy[-1] if hierarchy else None,
            "chunk_id": chunk_id,
        }
        metadata.update(extra_metadata)
        redacted_content = redact_pii(content)
        return Document(page_content=redacted_content, metadata=metadata)

    # Add all pages to current group
    current_group = elements
    flush_group()
    add_sequential_links(documents)

    return documents


def load_and_split_document(filepath: str):
    """Dispatcher: pick the right splitting strategy based on file type."""
    ext = Path(filepath).suffix.lower()
    if ext in [".docx", ".doc", ".txt"]:
        return split_textual_document(filepath)
    elif ext == ".pdf":
        return split_pdf_document(filepath)
    elif ext in [".csv", ".xlsx"]:
        return split_tabular_document(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
# ----------------------------
# Save and index documents
# ----------------------------
def save_and_index(project_id: int, doc_id: int, doc_name: str, filepath: str, doc_type: str, documents):
    pi = load_project_index(project_id)
    conn_thread = get_connection()
    c_thread = conn_thread.cursor()
    try:
        c_thread.execute("UPDATE documents SET status=? WHERE id=?", ("indexing", doc_id))
        if not pi.vectorstore:
            pi.vectorstore = FAISS.from_documents(documents, embedding_model)
        else:
            pi.vectorstore.add_documents(documents)
        for idx, doc in enumerate(documents, start=1):
            c_thread.execute(
    "INSERT INTO doc_chunks (doc_id, chunk_type, content, metadata) VALUES (?, ?, ?, ?)",
    (
        doc_id,
        doc.metadata.get("category")or "text_chunk",
        doc.page_content,
        json.dumps(doc.metadata, ensure_ascii=False)
    )
)
            progress_percent = int((idx / len(documents)) * 100)
            c_thread.execute("UPDATE documents SET progress=? WHERE id=?", (progress_percent, doc_id))
            if idx % 20 == 0 or idx == len(documents):
                conn_thread.commit()
        c_thread.execute("UPDATE documents SET status=?, progress=? WHERE id=?", ("completed", 100, doc_id))
        conn_thread.commit()
        persist_project_index(project_id)
    except Exception as e:
        print(f"Indexing error for doc {doc_id} in project {project_id}: {e}")
        c_thread.execute("UPDATE documents SET status=? WHERE id=?", ("error", doc_id))
        conn_thread.commit()
    finally:
        conn_thread.close()

# ----------------------------
# Rebuild FAISS after deletion
# ----------------------------
def rebuild_project_index_from_store(project_id: int):
    pi = load_project_index(project_id)
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
SELECT content, metadata 
FROM doc_chunks 
WHERE doc_id IN (SELECT id FROM documents WHERE project_id=?)
""", (project_id,))
    rows = c.fetchall()
    conn.close()
    if rows:
        documents = [Document(page_content=r[0], metadata=json.loads(r[1] or "{}")) for r in rows]
        pi.vectorstore = FAISS.from_documents(documents, embedding_model)
        persist_project_index(project_id)

# ----------------------------
# FastAPI app
# ----------------------------

from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)

# ----------------------------
# API models
# ----------------------------
class ProjectRequest(BaseModel):
    project_name: str

class QueryRequest(BaseModel):
    project_id: int
    query: str
    top_k: int = 5

# ----------------------------
# Project endpoints
# ----------------------------
@app.post("/project")
def create_or_get_project(req: ProjectRequest, current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id FROM projects WHERE name=?", (req.project_name,))
    row = c.fetchone()
    if row:
        conn.close()
        raise HTTPException(
            status_code=400,
            detail="❌ Project name already exists. Please choose a different name."
        )
        
    c.execute("INSERT INTO projects (name, created_by) VALUES (?, ?)", (req.project_name, user_id))
    pid = c.lastrowid
    conn.commit()
    # Get the user's role from users table
    c.execute("SELECT role FROM users WHERE id=?", (user_id,))
    user_role_row = c.fetchone()
    user_role = user_role_row[0] if user_role_row else "user"

    # Add entry to user_project table
    c.execute(
        "INSERT OR IGNORE INTO user_project (user_id, project_id, role) VALUES (?, ?, ?)",
        (user_id, pid, user_role)
    )
    conn.commit()
    conn.close()
    load_project_index(pid)
    return {"project_id": pid, "project_name": req.project_name, "user_id": user_id}

@app.get("/projects")
def list_projects(current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    conn = get_connection()
    c = conn.cursor()
    if current_user["role"] == "admin":
        c.execute("SELECT id, name, created_at FROM projects ORDER BY created_at DESC")
    else:
        c.execute("""
            SELECT p.id, p.name, p.created_at
            FROM projects p
            JOIN user_project up ON p.id = up.project_id
            WHERE up.user_id = ?
            ORDER BY p.created_at DESC
        """, (current_user["user_id"],))
    rows = [{"project_id": r[0], "project_name": r[1], "created_at": r[2]} for r in c.fetchall()]
    conn.close()
    return rows

@app.get("/project/{project_id}")
def project_details(project_id: int):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, name, path, type, status, progress FROM documents WHERE project_id=?", (project_id,))
    docs = [{"doc_id": r[0], "doc_name": r[1], "path": r[2], "type": r[3], "status": r[4], "progress": r[5]} for r in c.fetchall()]
    conn.close()
    return {"project_id": project_id, "documents": docs}

# ----------------------------
# Upload endpoint
# ----------------------------
@app.post("/upload")
async def upload_files(project_id: int = Query(...), files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = None, current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    conn_main = get_connection()
    c_main = conn_main.cursor()
    c_main.execute("SELECT id FROM projects WHERE id=?", (project_id,))
    if not c_main.fetchone():
        conn_main.close()
        return JSONResponse(status_code=404, content={"status": "error", "message": "Project not found"})
    conn_main.close()

    results = []
    proj_dir = os.path.join(RAW_DIR, str(project_id))
    os.makedirs(proj_dir, exist_ok=True)
    
    for file in files:
        original_name = file.filename
        filename = original_name
        filepath = os.path.join(proj_dir, filename)

        # Automatically rename if file already exists
        counter = 1
        name, ext = os.path.splitext(original_name)
        while os.path.exists(filepath):
            filename = f"{name} ({counter}){ext}"
            filepath = os.path.join(proj_dir, filename)
            counter += 1

        try:
            with open(filepath, "wb") as f:
                shutil.copyfileobj(file.file, f)

            conn_thread = get_connection()
            c_thread = conn_thread.cursor()
            c_thread.execute(
    "INSERT INTO documents (project_id, name, path, type, status, uploaded_by) VALUES (?, ?, ?, ?, ?, ?)",
    (project_id, filename, filepath, "pending", "pending", user_id)
)
            doc_id = c_thread.lastrowid
            conn_thread.commit()
            conn_thread.close()

            documents = load_and_split_document(filepath)

            if background_tasks:
                background_tasks.add_task(save_and_index, project_id, doc_id, filename, filepath, "doc", documents)
            else:
                save_and_index(project_id, doc_id, filename, filepath, "doc", documents)

            results.append({
                "status": "processing",
                "doc_id": doc_id,
                "doc_name": filename,
                "chunks": len(documents)
            })

        except Exception as e:
            results.append({"status": "error", "doc_name": filename, "message": str(e)})

    return results

# ----------------------------
# Status endpoint
# ----------------------------
@app.get("/status/{doc_id}")
async def check_status(doc_id: int):
    conn_thread = get_connection()
    c_thread = conn_thread.cursor()
    c_thread.execute("SELECT status, progress FROM documents WHERE id=?", (doc_id,))
    row = c_thread.fetchone()
    conn_thread.close()
    if row:
        return {"doc_id": doc_id, "status": row[0], "progress": row[1]}
    else:
        return {"doc_id": doc_id, "status": "not_found", "progress": 0}

# ----------------------------
# List documents for a project
# ----------------------------
@app.get("/project/{project_id}/documents")
def list_project_documents(project_id: int):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, name, path, type, status, progress FROM documents WHERE project_id=?", (project_id,))
    items = [{"doc_id": r[0], "doc_name": r[1], "path": r[2], "type": r[3], "status": r[4], "progress": r[5]} for r in c.fetchall()]
    conn.close()
    return {"project_id": project_id, "documents": items}

# ----------------------------
# Delete a document
# ----------------------------
@app.delete("/project/{project_id}/documents/{doc_id}")
def delete_document(project_id: int, doc_id: int):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, path FROM documents WHERE id=? AND project_id=?", (doc_id, project_id))
    r = c.fetchone()
    if not r:
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found in project")

    try:
        c.execute("DELETE FROM doc_chunks WHERE doc_id=?", (doc_id,))
        c.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        conn.commit()
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Failed to delete document rows: {e}")
    conn.close()

    try:
        if os.path.exists(r[1]):
            os.remove(r[1])
    except Exception:
        pass

    pi = load_project_index(project_id)
    rebuild_project_index_from_store(project_id)
    return {"status": "deleted", "doc_id": doc_id}

def load_data():
    """
    Load data from SQLite database for analytics dashboard
    """
    conn = sqlite3.connect(DB_PATH)
    queries = pd.read_sql("SELECT * FROM queries", conn)
    feedback = pd.read_sql("SELECT * FROM feedback", conn)
    projects = pd.read_sql("SELECT * FROM projects", conn)
    documents = pd.read_sql("SELECT * FROM documents", conn)
    users = pd.read_sql("SELECT * FROM users", conn)
    conn.close()
    return queries, feedback, projects, documents, users


# ----------------------------
# GEMINI LLM
# ----------------------------

def call_gemini(prompt: str, retries: int = 3, base_delay: int = 2, stream: bool = False):
    """
    Call Gemini API with fallback models and optional streaming.
    If stream=True → yields partial chunks instead of returning full text.
    """
    models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    attempt_count = 0
    is_fallback = False
    for model in models:
        for attempt in range(1, retries + 2):
            attempt_count += 1
            try:
                # --- STREAMING MODE ---
                if stream:
                    response_stream = genai_client.models.generate_content_stream(
                        model=model,
                        contents=prompt,
                    )
                    for chunk in response_stream:
                        if hasattr(chunk, "text") and chunk.text:
                            yield {
                                "text": chunk.text,
                                "attempt_count": attempt_count,
                                "is_fallback": is_fallback
                            }  # Send to frontend
                    return  # stop after yielding stream

                # --- NON-STREAMING (original path) ---
                response = genai_client.models.generate_content(model=model, contents=prompt)
                return (
                    getattr(response, "text", None)
                    or getattr(response, "content", None)
                    or str(response),
                    attempt_count,
                    False,
                )

            except Exception as e:
                err_msg = str(e)
                logging.error(f"[Gemini API Error] Model={model} Attempt={attempt}: {err_msg}")
                if "503" in err_msg or "overloaded" in err_msg.lower():
                    if attempt <= retries:
                        delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                        time.sleep(delay)
                        continue
                    else:
                        is_fallback = True
                        break
                else:
                    is_fallback = True
                    if stream:
                        yield {
                            "text": "❌ Something went wrong while processing your request.",
                            "attempt_count": attempt_count,
                            "is_fallback": True
                        }
                        return
                    return "❌ Something went wrong while processing your request.", attempt_count, True

    if stream:
        yield {
            "text": "⚠️ All Gemini models are busy. Please try again later.",
            "attempt_count": attempt_count,
            "is_fallback": True
        }
        return
    return "⚠️ All Gemini models are busy. Please try again later.", attempt_count, True

# ---------------------------------------
# Document Detection in the User's Query
# ---------------------------------------

import re

def extract_doc_names_from_query(project_id: int,query: str) -> List[str]:
    """
    Detect document names mentioned in the user query.
    Simple heuristic: look for words matching existing document names in the project.
    """
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT name FROM documents WHERE project_id=?", (project_id,))
    doc_names = [r[0] for r in c.fetchall()]
    conn.close()
    
    # Case-insensitive match of any doc name in the query
    mentioned = [name for name in doc_names if re.search(rf"\b{name}\b", query, re.IGNORECASE)]
    return mentioned

# ---------------------------------------------
# Retrieval based on Document in the User Query
# ---------------------------------------------


def retrieve_from_doc_by_name(project_id: int, doc_name: str) -> List[Document]:
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id FROM documents WHERE project_id=? AND LOWER(name)=?", (project_id, doc_name.lower()))
    row = c.fetchone()
    if not row:
        conn.close()
        return []
    doc_id = row[0]
    c.execute("SELECT content, metadata FROM doc_chunks WHERE doc_id=?", (doc_id,))
    rows = c.fetchall()
    conn.close()

    docs = []
    for content, metadata_json in rows:
        metadata = json.loads(metadata_json) if metadata_json else {}
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

# ----------------------------
# Retrieval based on User's Query
# ----------------------------
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List

def retrieve_relevant_chunks(project_id: int, query: str, threshold: float = 0.7, max_candidates: int = 50, alpha: float = 0.5) -> List[Document]:
    """
    Hybrid retrieval using FAISS (semantic) + BM25 (keyword) with relational awareness:
    - Pull top-N candidates from FAISS.
    - Compute BM25 scores on all docs.
    - Fuse scores.
    - Include related chunks using group_id and sequential links.
    - Deduplicate and return ranked results.
    """

    def normalize_key(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip()).lower()
    
    pi = load_project_index(project_id)
    if not pi.vectorstore:
        return []

    # ----------------------------
    # Step 1: Embed query for FAISS
    # ----------------------------
    query_embedding = pi.vectorstore.embedding_function.embed_query(query)
    query_embedding = np.array([query_embedding]).astype("float32")

    # ----------------------------
    # Step 2: FAISS retrieval
    # ----------------------------
    faiss_index = pi.vectorstore.index
    scores, indices = faiss_index.search(query_embedding, max_candidates)

    faiss_docs = []
    faiss_scores = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        doc_id = pi.vectorstore.index_to_docstore_id[idx]
        doc = pi.vectorstore.docstore.search(doc_id)
        if not doc or not getattr(doc, "page_content", "").strip():
            continue
        faiss_docs.append(doc)
        faiss_scores.append(score)

    # ----------------------------
    # Step 3: BM25 retrieval
    # ----------------------------
    all_docs = [d for d in pi.vectorstore.docstore._dict.values() if getattr(d, "page_content", "").strip()]
    tokenized_corpus = [d.page_content.split() for d in all_docs]
    bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
    bm25_scores = bm25.get_scores(query.split()) if bm25 else [0] * len(all_docs)

    # ----------------------------
    # Step 4: Normalize scores
    # ----------------------------
    def normalize(arr):
        arr = np.array(arr)
        if arr.max() == arr.min():
            return np.ones_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min())

    faiss_scores_norm = normalize(np.array(faiss_scores)) if faiss_scores else np.array([])
    bm25_scores_norm = normalize(np.array(bm25_scores)) if bm25_scores is not None and bm25_scores.size > 0 else np.array([])


    # ----------------------------
    # Step 5: Fuse scores with relational awareness
    # ----------------------------
    final_scores = {}

    def add_doc(doc, score):
        key = normalize_key(doc.page_content)
        if key in final_scores:
            final_scores[key] += score
        else:
            final_scores[key] = score

    # FAISS top docs
    for doc, f_score in zip(faiss_docs, faiss_scores_norm):
        add_doc(doc, alpha * f_score)

        # Add related group chunks
        if "group_id" in doc.metadata:
            group_id = doc.metadata["group_id"]
            for d in all_docs:
                if d.metadata.get("group_id") == group_id and d.page_content != doc.page_content:
                    add_doc(d, alpha * f_score * 0.8)  # slightly lower weight for related

        # Add sequential neighbors
        prev_id = doc.metadata.get("prev_chunk_id")
        next_id = doc.metadata.get("next_chunk_id")
        for neighbor in all_docs:
            if neighbor.metadata.get("chunk_id") in [prev_id, next_id]:
                add_doc(neighbor, alpha * f_score * 0.6)

    # BM25 scores
    for doc, b_score in zip(all_docs, bm25_scores_norm):
        add_doc(doc, (1 - alpha) * b_score)

    # ----------------------------
    # Step 6: Rank & Deduplicate
    # ----------------------------
    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    unique_docs, seen = [], set()
    for content, _ in ranked:
        key = normalize_key(content)
        if key not in seen:
            for d in all_docs:
                if normalize_key(d.page_content) == key:
                    unique_docs.append(d)
                    break
            seen.add(key)
            
    return unique_docs[:max_candidates]


# ----------------------------
# Query WorkFlow
# ----------------------------

from typing import Generator

def query_workflow(project_id: int, user_query: str) -> Generator[dict, None, dict]:
    """
    Dynamic query-driven workflow:
    - Detects mentioned documents in the user query.
    - Retrieves relevant chunks from those documents and the overall project.
    - Sends structured context to Gemini LLM to perform the requested actions.
    """
    
    # Step 1: Detect document names mentioned in the query
    mentioned_docs = extract_doc_names_from_query(project_id,user_query)

    # Step 2: Retrieve chunks for mentioned documents first
    contexts = []
    for doc in mentioned_docs:
        doc_chunks = retrieve_from_doc_by_name(project_id, doc)
        contexts.extend(doc_chunks)
    
    # Step 3: Retrieve additional project-wide relevant chunks via FAISS
    project_chunks = retrieve_relevant_chunks(project_id, user_query)
    contexts.extend(project_chunks)
    
    seen_texts = set()
    unique_contexts = []
    for doc in contexts:
        if doc.page_content not in seen_texts:
            unique_contexts.append(doc)
            seen_texts.add(doc.page_content)
    
    context_json = []
    for chunk in unique_contexts:
        context_json.append({
            "content": chunk.page_content,
            "source": chunk.metadata.get("filename", "unknown"),
            "section": chunk.metadata.get("hierarchy", ""),
            "chunk_index": chunk.metadata.get("chunk_index", 0)
        })
        
    #yield {"contexts": context_json}   
    final_json = json.dumps(context_json, separators=(",", ":"))
    structured_prompt = f"""
    # ROLE: Expert SQL Assistant
    # CONTEXT: You are analyzing documentation to generate precise SQL queries.

    # AVAILABLE DOCUMENTATION:
    {final_json}

    # USER QUERY TO ANSWER:
    {user_query}

    # CRITICAL INSTRUCTIONS (MUST FOLLOW):

    1. Generate valid SQL queries that answer the user question exactly.
    2. Use proper table and column names as given in the contexts.
    3. ***CRITICAL: Carefully read and interpret the content and their relationships. This point must be strictly followed.***
    - Extract all tables, columns, datatypes, PKs, FKs, and relationships
    - Infer business rules or constraints when mentioned
    4. Provide a complete, coherent, and accurate response based only on the above contexts.
    5. SQL queries should be in copy-friendly format within "```sql ```".
    6. When explaining the query:
    - generate an ER diagram-like ASCII structure in separate copy-friendly format:
    - Each table in its own box (┌─┐ │ └─┘)
    - PKs with "PK:" prefix, FKs with "[FK]"
    - Show joins with ◄────► arrows
    - Place aggregation/filter/GROUP BY/HAVING/ORDER BY/other conditions boxes separately and join them.
    - Keep it compact and aligned, like a schema diagram.
    7. Use proper indentation for readability.
    """
    

    # Step 5: Call Gemini API and return the answer
    # --- Yield Gemini stream ---
    # Step 4: Call Gemini API streaming
    last_attempt_count = 0
    last_is_fallback = False
    for chunk in call_gemini(structured_prompt, stream=True):
        last_attempt_count = chunk.get("attempt_count", last_attempt_count)
        last_is_fallback = chunk.get("is_fallback", last_is_fallback)
        yield {"text": chunk.get("text", ""), "attempt_count": last_attempt_count, "is_fallback": last_is_fallback, "contexts": final_json}

    # Step 5: Return contexts safely as attribute in workflow dict (not function)
    return {"contexts": final_json, "attempt_count": last_attempt_count, "is_fallback": last_is_fallback}



def evaluate_query_rag(project_id, user_query, sql_gen, docs):
    import sqlglot
    import re
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Ensure docs is a list
    if not isinstance(docs, list):
        docs = [docs]

    # Normalize all docs to dicts with 'content'
    normalized_docs = []
    for d in docs:
        if hasattr(d, "page_content"):  # Document object
            normalized_docs.append({"content": d.page_content})
        elif isinstance(d, dict):
            if "page_content" in d:
                normalized_docs.append({"content": d["page_content"]})
            elif "content" in d:
                normalized_docs.append({"content": d["content"]})
            else:
                normalized_docs.append({"content": str(d)})
        elif isinstance(d, str):
            normalized_docs.append({"content": d})
        else:
            normalized_docs.append({"content": str(d)})
            
    if not normalized_docs:
        normalized_docs = [{"content": ""}]


    # --- Extract SQL from sql blocks ---
    def extract_sql_blocks(text):
        pattern = r"```sql(.*?)```"
        blocks = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if not blocks:
            # fallback: extract lines starting with SQL keywords
            blocks = re.findall(r"(SELECT .*?;)", text, flags=re.DOTALL | re.IGNORECASE)
        return [b.strip() for b in blocks]

    # --- Extract SQL claims ---
    def extract_sql_claims(sql_text):
        import re
        import sqlglot
        from sqlglot.expressions import Table, Column, Join

        claims = []
        trees = []

        # Split into statements by semicolon
        statements = [stmt.strip() for stmt in sql_text.split(";") if stmt.strip()]

        for stmt in statements:
            try:
                parsed_trees = sqlglot.parse(stmt)
            except Exception as e:
                claims.append(f"parse_error:{e}")
                continue

            for tree in parsed_trees:
                if tree is None:   # 🔥 Skip empty parse results
                    continue
                trees.append(tree)

                # Extract tables
                for t in tree.find_all(Table):
                    if t.name:
                        claims.append(f"table:{t.name.lower()}")

                # Extract columns
                for c in tree.find_all(Column):
                    if c.name:
                        col_name = c.name.lower().split(".")[-1]
                        claims.append(f"column:{col_name}")

                # Extract joins
                for j in tree.find_all(Join):
                    try:
                        on_expr = j.on() if callable(j.on) else j.on
                        if on_expr:
                            join_clause = re.sub(r"\s+", " ", on_expr.sql().lower()).strip()
                            claims.append(f"join:{join_clause}")
                    except Exception:
                        # safely ignore joins that fail
                        continue

        return claims, trees


    # --- Combine all SQL blocks ---
    sql_blocks = extract_sql_blocks(sql_gen)
    if not sql_blocks:
        print("[RAG DEBUG] No SQL blocks found in answer.")
    else:
        print("[RAG DEBUG] SQL Blocks Found:", sql_blocks)
    all_claims = []
    all_trees = []
    for block in sql_blocks:
        claims, trees = extract_sql_claims(block)
        all_claims.extend(claims)
        all_trees.extend(trees)

    # --- Claim support rate ---
    def claim_support_rate(claims, normalized_docs):
        if not claims or not normalized_docs:
            return 0.0
        supported = 0
        for c in claims:
            if ":" not in c:
                continue
            claim_text = c.split(":", 1)[-1].lower()
            found = any(claim_text in d["content"].lower() for d in normalized_docs)
            if found:
                supported += 1
        return supported / len(claims)

    # --- Precision @ K ---
    def precision_at_k(normalized_docs, claims, k=5):
        if not claims or not normalized_docs:
            return 0.0
        claim_names = [c.split(":", 1)[-1].lower() for c in claims if ":" in c]
        relevant_count = 0
        top_docs = normalized_docs[:k]
        for d in top_docs:
            content_lower = d["content"].lower()
            if any(cn in content_lower for cn in claim_names):
                relevant_count += 1
        return relevant_count / k

    # --- Static SQL execution score ---
    from sqlglot.expressions import Table, Column

    def static_sql_score(sql_trees, normalized_docs):
        if not sql_trees:
            return 0.0

        scores = []

        for tree in sql_trees:
            if tree is None:   # 🔥 Guard here too
                continue
            try:
                score_parts = []

                # Columns
                cols = [c.name.lower().split(".")[-1] for c in tree.find_all(Column)]
                col_hits = sum(any(col in d["content"].lower() for d in normalized_docs) for col in cols)
                score_parts.append(col_hits / len(cols) if cols else 1.0)

                # Tables
                tables = [t.name.lower() for t in tree.find_all(Table)]
                table_hits = sum(any(t in d["content"].lower() for d in normalized_docs) for t in tables)
                score_parts.append(table_hits / len(tables) if tables else 1.0)

                scores.append(sum(score_parts) / len(score_parts) if score_parts else 1.0)
                
            except Exception as e:
                print(f"⚠️ Error scoring tree: {e}")
                continue

        return sum(scores) / len(scores)

    # --- Query coverage using user query dynamically ---
    def query_coverage(user_query, normalized_docs):
        if not user_query or not normalized_docs:
            return 0.0
        query_terms = [w.lower() for w in re.findall(r"\b\w+\b", user_query) if len(w) > 2]
        if not query_terms:
            return 0.0
        covered = 0
        for term in query_terms:
            found = any(term in d["content"].lower() for d in normalized_docs)
            if found:
                covered += 1
        return covered / len(query_terms)

    # --- Debug ---
    print("Top 3 chunk texts:", [d["content"][:100] for d in normalized_docs[:3]])
    print("Extracted SQL claims:", all_claims)

    if all_claims and all_trees:
        precision1 = precision_at_k(normalized_docs, all_claims, 1)
        precision3 = precision_at_k(normalized_docs, all_claims, 3)
        precision5 = precision_at_k(normalized_docs, all_claims, 5)
        claim_support = claim_support_rate(all_claims, normalized_docs)
        static_exec = static_sql_score(all_trees, normalized_docs)
    else:
        # No SQL found, don't penalize
        precision1 = precision3 = precision5 = None
        claim_support = static_exec = None

    # Query coverage can still be computed, because it's based on the user query
    query_cov = query_coverage(user_query, normalized_docs)

    return {
        "Precision@1": precision1,
        "Precision@3": precision3,
        "Precision@5": precision5,
        "ClaimSupportRate": claim_support,
        "StaticExecScore": static_exec,
        "QueryCoverage": query_cov,
        "NumClaims": len(all_claims)
    }



# ----------------------------
# Chat Model
# ----------------------------

class ChatRequest(BaseModel):
    project_id: int
    query: str
    conversation_id: Optional[str] = None  # optional
    top_k: int = 5


@app.post("/chat")
async def chat(req: ChatRequest, current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    conversation_id = req.conversation_id or str(uuid.uuid4())
    start_time = time.time()
    safe_query = redact_pii(req.query)
    safe_query = sanitize_user_query(safe_query)
    
    # --- memory initialization (same as before) ---
    if conversation_id not in conversation_memory:
        conversation_memory[conversation_id] = []
        conn = get_connection()
        c = conn.cursor()
        c.execute("""
            SELECT query, answer FROM queries
            WHERE conversation_id=? AND project_id=? AND user_id=?
            ORDER BY created_at
        """, (conversation_id, req.project_id, user_id))
        rows = c.fetchall()
        conn.close()
        for q, a in rows:
            conversation_memory[conversation_id].append({"role": "user", "content": q})
            conversation_memory[conversation_id].append({"role": "assistant", "content": a})

    memory = conversation_memory[conversation_id]
    memory.append({"role": "user", "content": safe_query})

    chat_history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in memory])
    prompt = f"{chat_history_str}\nAssistant:"

    # --- Initialize a placeholder in DB if needed ---
    query_id = str(uuid.uuid4())
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO queries (id, project_id, query, answer, user_id, conversation_id, response_time)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (query_id, req.project_id, safe_query, "", user_id, conversation_id, 0))
    conn.commit()
    conn.close()

    # --- Streaming generator ---
    async def stream_and_save():
        full_answer = ""
        attempt_count = 0
        is_fallback = False
        contexts = ""
        for chunk in query_workflow(req.project_id, prompt):
            text = chunk.get("text", "")
            full_answer += text

            # Update attempt/fallback
            attempt_count = chunk.get("attempt_count", attempt_count)
            is_fallback = chunk.get("is_fallback", is_fallback)
            ctx = chunk.get("contexts", contexts)

            # ✅ Normalize context type
            if isinstance(ctx, str):
                try:
                    ctx = json.loads(ctx)
                except:
                    ctx = []
            contexts = ctx

            yield text
        

        # --- After streaming finishes ---
        response_time = time.time() - start_time

        # Append to memory
        memory.append({"role": "assistant", "content": full_answer})
        if len(memory) > MAX_MEMORY_LENGTH:
            conversation_memory[conversation_id] = memory[-MAX_MEMORY_LENGTH:]

        # --- Compute metrics (optional) ---
        pi = load_project_index(req.project_id)
        print(contexts)
        metrics = evaluate_query_rag(req.project_id, safe_query, full_answer, contexts)
        print(f"[RAG Metrics] Query: {safe_query}\n{metrics}\n")

        # --- Update DB with final answer and metrics ---
        conn = get_connection()
        c = conn.cursor()
        c.execute("""
            UPDATE queries
            SET answer = ?, response_time = ?, attempts = ?, is_fallback = ?, 
                precision1 = ?, precision3 = ?, precision5 = ?, 
                claim_support = ?, static_exec = ?, query_coverage = ?, num_claims = ?
            WHERE id = ?
        """, (
            full_answer,
            response_time,
            attempt_count,  # or attempt_count if available
            is_fallback,  # or is_fallback if available
            metrics.get("Precision@1"),
            metrics.get("Precision@3"),
            metrics.get("Precision@5"),
            metrics.get("ClaimSupportRate"),
            metrics.get("StaticExecScore"),
            metrics.get("QueryCoverage"),
            metrics.get("NumClaims"),
            query_id
        ))
        conn.commit()
        conn.close()
        
        yield f"\n[[[META]]]{json.dumps({'contexts': contexts, 'query_id': query_id})}"

    return StreamingResponse(stream_and_save(), media_type="text/plain")
    


# ----------------------------
# Feedback model
# ----------------------------
class Feedback(BaseModel):
    query_id: str
    project_id: int
    query: str = ""
    answer: str
    contexts: Any = []
    feedback: Literal["up", "down"]   # "up" or "down"
    comment: str = ""
    user_id: str

# ----------------------------
# Load existing feedback
# ----------------------------
if os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "r") as f:
        feedback_store = json.load(f)
else:
    feedback_store = []


# ----------------------------
# Feedback endpoint
# ----------------------------
@app.post("/feedback")
async def submit_feedback(item: Feedback, current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    try:
        contexts = item.contexts
        if isinstance(contexts, str):
            try:
                contexts = json.loads(contexts)  # parse JSON string
            except json.JSONDecodeError:
                contexts = [contexts]  # wrap raw string in list
        elif contexts is None:
            contexts = []
        elif not isinstance(contexts, list):
            contexts = [contexts]  # wrap anything else in list

        item.contexts = contexts
        # --- Load existing feedback ---
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "r") as f:
                feedback_store = json.load(f)
        else:
            feedback_store = []

        # --- Save new feedback entry ---
        fb_entry = {
            "id": str(uuid.uuid4()),
            "query_id": item.query_id,
            "project_id": item.project_id,
            "query": item.query,
            "answer": item.answer,
            "contexts": item.contexts,
            "feedback": item.feedback,
            "comment": item.comment,
            "user_id": user_id
        }
        feedback_store.append(fb_entry)

        with open(FEEDBACK_FILE, "w") as f:
            json.dump(feedback_store, f, indent=2)

        # --- Save feedback to SQLite ---
        conn = get_connection()
        c = conn.cursor()
        c.execute("""
            INSERT INTO feedback (id, query_id, project_id, answer, contexts, feedback, comment, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fb_entry["id"],
            fb_entry["query_id"],
            fb_entry["project_id"],
            fb_entry["answer"],
            json.dumps(fb_entry["contexts"], separators=(",", ":")),
            fb_entry["feedback"],
            fb_entry["comment"],
            fb_entry["user_id"]
        ))
        conn.commit()
        conn.close()

        return {
            "status": "success",
            "message": "Feedback stored",
            "id": fb_entry["id"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing feedback: {e}")
    
# ----------------------------
# API Endpoint: Get all conversations/queries
# ----------------------------
@app.get("/queryhistory/{project_id}")
def query_history(project_id: str, current_user: dict = Depends(get_current_user)):
    """
    Fetch all queries (conversation history) for the admin/viewing purpose.
    This allows users to see past conversations.
    """
    conn = get_connection()
    c = conn.cursor()
    user_id = current_user["user_id"]
    try:
        # Fetch all queries from the database
        c.execute(
           """
            SELECT id, conversation_id, project_id, query, answer, user_id, response_time, created_at
            FROM queries
            WHERE user_id = ? AND project_id = ?
            ORDER BY created_at DESC
            """,
            (user_id, project_id),
        )
        rows = c.fetchall()

        conversations = [
            {
                "id": row[0],
                "conversation_id": row[1],
                "project_id": row[2],
                "query": row[3],
                "answer": row[4],
                "user_id": row[5],
                "response_time": row[6],
                "created_at": row[7],
            }
            for row in rows
        ]

        return conversations

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching conversations: {e}"
        )
    finally:
        conn.close()


# ----------------------------
# Run with uvicorn
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_backend:app", host="127.0.0.1", port=8000, reload=True)
