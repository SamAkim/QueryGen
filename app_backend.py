import io
import os
import sqlite3
import uuid
from fastapi.staticfiles import StaticFiles
import fitz
import pandas as pd
import numpy as np
import pickle
import json
import random
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, List, Dict, Literal, Optional, TypedDict, Union
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
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
import jira
from jira import JIRA
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

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

def init_db():
    conn = get_connection()
    c = conn.cursor()

    # ---------------------------------------------------------
    # 1. Create Base Tables
    # ---------------------------------------------------------

    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,  -- UUID
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        username TEXT UNIQUE,
        password_hash TEXT,
        role TEXT DEFAULT 'user'  -- 'admin' or 'user'
    )
    """)

    # Base definition of queries table
    c.execute("""
    CREATE TABLE IF NOT EXISTS queries (
        id TEXT PRIMARY KEY,
        project_id INTEGER,
        query TEXT,
        answer TEXT,
        user_id TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        conversation_id TEXT,
        response_time FLOAT,
        attempts INTEGER DEFAULT 1,
        is_fallback BOOLEAN DEFAULT 0,
        precision1 REAL,
        precision3 REAL,
        precision5 REAL,
        claim_support REAL,
        static_exec REAL,
        query_coverage REAL,
        num_claims INTEGER,
        database_engine TEXT DEFAULT 'postgresql'
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

    c.execute("""
    CREATE TABLE IF NOT EXISTS user_project (
        user_id TEXT,
        project_id INTEGER,
        role TEXT DEFAULT 'user',
        PRIMARY KEY(user_id, project_id)
    )
    """)

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
        project_id INTEGER,
        query TEXT,
        answer TEXT,
        contexts TEXT,
        feedback TEXT,
        comment TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        user_id TEXT,
        query_id TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS jira_connections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        jira_url TEXT,
        username TEXT,
        api_token TEXT,
        is_default BOOLEAN DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS jira_imported_issues (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER,
        jira_issue_key TEXT,
        jira_issue_summary TEXT,
        jira_issue_description TEXT,
        jira_issue_type TEXT,
        jira_issue_status TEXT,
        jira_issue_assignee TEXT,
        jira_issue_attachments TEXT,  -- JSON array of attachment info
        imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (project_id) REFERENCES projects (id)
    )
    """)

    # ---------------------------------------------------------
    # 2. Migration Logic (Update columns if missing)
    # ---------------------------------------------------------
    
    # Check if chain_id exists in queries table
    try:
        c.execute("SELECT chain_id FROM queries LIMIT 1")
    except Exception:
        print("🔧 Migrating DB: Adding chain_id and version columns to 'queries' table...")
        
        # Add chain_id column
        try:
            c.execute("ALTER TABLE queries ADD COLUMN chain_id TEXT")
            print("   - Added column: chain_id")
        except Exception as e: 
            print(f"   - Migration note (chain_id): {e}")
        
        # Add version column
        try:
            c.execute("ALTER TABLE queries ADD COLUMN version INTEGER DEFAULT 1")
            print("   - Added column: version")
        except Exception as e: 
            print(f"   - Migration note (version): {e}")

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
        "PHONE": r"""
            (?:(?:\+?\d{1,3}[\s\-]?)?       # Optional country code
            \(?\d{2,4}\)?[\s\-\.]?)?        # Optional area code
            \d{2,4}[\s\-\.]?\d{3,4}         # Main number
            [\s\-\.]?\d{0,4}                # Optional extension
            (?!\d*-\d*-\d*)                 # Negative lookahead for dates
        """,
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
    filename = Path(filepath).name
    for sheet, df in sheets.items():
        # Schema / header info
        documents.append(Document(
            page_content=f"Sheet: {sheet}\nColumns: {', '.join(df.columns)}",
            metadata={"category": "schema", "sheet": sheet, "filename": filename}
        ))

        # Row batches
        for start in range(0, len(df), batch_size):
            batch = df.iloc[start:start + batch_size]
            batch_text = f"Sheet: {sheet}\nRows {start}-{start+len(batch)-1}:\n{batch.to_csv(index=False)}"
            
            redacted_content = redact_pii(batch_text)
            
            documents.append(Document(
                page_content=redacted_content,
                metadata={"category": "rows", "sheet": sheet, "batch": f"{start}-{start+len(batch)-1}", "filename": filename}
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
    "http://localhost:3000",
    "http://localhost:8000"
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
    
    c.execute("SELECT id FROM users WHERE role='admin'")
    admin_rows = c.fetchall()

    for admin in admin_rows:
        admin_id = admin[0]
        c.execute(
            "INSERT OR IGNORE INTO user_project (user_id, project_id, role) VALUES (?, ?, ?)",
            (admin_id, pid, 'admin')
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
            
        # Normalize to absolute + forward slashes
        filepath = os.path.abspath(filepath).replace("\\", "/")

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
# Download/Open a document
# ----------------------------
@app.get("/project/{project_id}/documents/{doc_id}/download")
def download_document(
    project_id: int, 
    doc_id: int, 
    current_user: dict = Depends(get_current_user)
):
    conn = get_connection()
    c = conn.cursor()
    
    # First, verify the user has access to this project
    user_id = current_user["user_id"]
    if current_user["role"] != "admin":
        c.execute("""
            SELECT 1 FROM user_project 
            WHERE user_id = ? AND project_id = ?
        """, (user_id, project_id))
        if not c.fetchone():
            conn.close()
            raise HTTPException(status_code=403, detail="Access denied to this project")
    
    c.execute("SELECT id, name, path FROM documents WHERE id=? AND project_id=?", (doc_id, project_id))
    r = c.fetchone()
    if not r:
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found in project")
    
    doc_id, doc_name, file_path = r
    conn.close()

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on server")
    
    # Return the file for download
    return FileResponse(
        path=file_path,
        filename=doc_name,
        media_type='application/octet-stream'
    )
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

    if project_id in project_cache:
        del project_cache[project_id]
        
    d = project_dir(project_id)
    vs_path = os.path.join(d, "faiss_store.pkl")
    if os.path.exists(vs_path):
        os.remove(vs_path)
        
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
from google.genai import types
from google.genai.types import (
    HttpOptions,
    Tool,
    ToolCodeExecution,
    GenerateContentConfig,
)

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
                response = genai_client.models.generate_content(
                    model=model, 
                    contents=prompt,
                    config=GenerateContentConfig(temperature=0.3)  # Lower temp
                )
                # Extract text from response
                if hasattr(response, 'text'):
                    return response.text, attempt_count, is_fallback
                elif hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        return "".join(part.text for part in candidate.content.parts if hasattr(part, 'text')), attempt_count, is_fallback
                    
                # Fallback to string conversion
                return str(response), attempt_count, is_fallback

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
                elif "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
                    print(f"🚫 [Rate Limit] Quota exceeded for {model}, attempt {attempt}")
                    if attempt <= retries:
                        # Use longer delays for rate limits (30s, 60s, 90s...)
                        delay = 10 * attempt + random.uniform(0, 5)
                        print(f"⏳ [Rate Limit] Waiting {delay:.2f}s before retry...")
                        time.sleep(delay)
                        continue
                    else:
                        is_fallback = True
                        break
                else:
                    is_fallback = True
                    return "❌ Something went wrong while processing your request.", attempt_count, True

    return "⚠️ All Gemini models are busy. Please try again later.", attempt_count, True


def call_llm(prompt: str, retries: int = 3, base_delay: int = 2) -> str:
    """
    Simple synchronous Gemini API call for planning and retrieval agents.
    Returns only the text response.
    """
    models = ["gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash"]

    for model in models:
        for attempt in range(1, retries + 2):
            try:
                response = genai_client.models.generate_content(
                    model=model, 
                    contents=prompt,
                    config=GenerateContentConfig(temperature=0.1)  # Lower temp for planning/retrieval
                )
                
                # Extract text from response
                if hasattr(response, 'text'):
                    return response.text
                elif hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        return "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                
                # Fallback to string conversion
                return str(response)
                
            except Exception as e:
                err_msg = str(e)
                logging.error(f"[Gemini API Error] Model={model} Attempt={attempt}: {err_msg}")
                if "503" in err_msg or "overloaded" in err_msg.lower():
                    if attempt <= retries:
                        delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                        time.sleep(delay)
                        continue
                    else:
                        break
                elif "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
                    print(f"🚫 [Rate Limit] Quota exceeded for {model}, attempt {attempt}")
                    if attempt <= retries:
                        # Use longer delays for rate limits (30s, 60s, 90s...)
                        delay = 10 * attempt + random.uniform(0, 5)
                        print(f"⏳ [Rate Limit] Waiting {delay:.2f}s before retry...")
                        time.sleep(delay)
                        continue
                    else:
                        break
                else:
                    break

    return "⚠️ All Gemini models are busy. Please try again later."



def get_project_documents(project_id: int) -> list:
    """Return a list of document names for the given project_id."""
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("SELECT name FROM documents WHERE project_id=?", (project_id,))
        rows = c.fetchall()
        conn.close()
        return [r[0] for r in rows] if rows else []
    except Exception as e:
        print(f"⚠️ Error fetching project documents: {e}")
        return []


# ---------------------------------------
# Document Detection in the User's Query
# ---------------------------------------

import re

def extract_doc_names_from_query(project_id: int, query: str) -> List[str]:
    """
    Detect document names mentioned in the user query with flexible matching.
    """
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT name FROM documents WHERE project_id=?", (project_id,))
    doc_names = [r[0] for r in c.fetchall()]
    conn.close()
    
    if not doc_names:
        return []
    
    # Normalize query for better matching
    query_lower = query.lower()
    
    mentioned = []
    for doc_name in doc_names:
        doc_name_lower = doc_name.lower()
        
        # Multiple matching strategies:
        # 1. Exact word match (original behavior)
        if re.search(rf"\b{re.escape(doc_name_lower)}\b", query_lower):
            mentioned.append(doc_name)
            continue
            
        # 2. Partial match (handles extensions, spaces, etc.)
        # Remove file extensions for matching
        doc_base = re.sub(r'\.(pdf|docx?|txt|csv|xlsx?)$', '', doc_name_lower)
        query_clean = re.sub(r'[^\w\s]', ' ', query_lower)  # Replace punctuation with spaces
        
        if doc_base in query_clean.split():
            mentioned.append(doc_name)
            continue
            
        # 3. Handle cases like "sample.pdf" → "sample"
        if re.search(rf"\b{re.escape(doc_base)}\b", query_lower):
            mentioned.append(doc_name)
    
    print(f"🔍 Document extraction: query='{query}' → found={mentioned}")
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

def retrieve_relevant_chunks(project_id: int, query: str, keyword_terms: List[str], threshold: float = 0.7, max_candidates: int = 50, alpha: float = 0.5) -> List[Document]:
    """
    Hybrid retrieval using FAISS (semantic) + BM25 (keyword) with relational awareness.
    """
    def normalize_key(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip()).lower()
    
    pi = load_project_index(project_id)
    if not pi.vectorstore:
        return []

    # Step 1: Embed query for FAISS
    query_embedding = pi.vectorstore.embedding_function.embed_query(query)
    query_embedding = np.array([query_embedding]).astype("float32")

    # Step 2: FAISS retrieval
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

    # Step 3: BM25 retrieval
    all_docs = [d for d in pi.vectorstore.docstore._dict.values() if getattr(d, "page_content", "").strip()]
    if keyword_terms:
        # Use only the extracted keywords for BM25
        keyword_query = " ".join(keyword_terms)
        print(f"🔍 BM25 using ONLY keywords: {keyword_terms}")
        
        tokenized_corpus = [d.page_content.split() for d in all_docs]
        bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        bm25_scores = bm25.get_scores(keyword_query.split()) if bm25 else [0] * len(all_docs)
    else:
        tokenized_corpus = [d.page_content.split() for d in all_docs]
        bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        bm25_scores = bm25.get_scores(query.split()) if bm25 else [0] * len(all_docs)

    # Step 4: Normalize scores
    def normalize(arr):
        arr = np.array(arr)
        if arr.max() == arr.min():
            return np.ones_like(arr)
        return (arr - arr.min()) / (arr.max() - arr.min())

    faiss_scores_norm = normalize(np.array(faiss_scores)) if faiss_scores else np.array([])
    bm25_scores_norm = normalize(np.array(bm25_scores)) if bm25_scores is not None and bm25_scores.size > 0 else np.array([])

    # Step 5: Fuse scores with relational awareness
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
                    add_doc(d, alpha * f_score * 0.8)

        # Add sequential neighbors
        prev_id = doc.metadata.get("prev_chunk_id")
        next_id = doc.metadata.get("next_chunk_id")
        for neighbor in all_docs:
            if neighbor.metadata.get("chunk_id") in [prev_id, next_id]:
                add_doc(neighbor, alpha * f_score * 0.6)

    # BM25 scores
    for doc, b_score in zip(all_docs, bm25_scores_norm):
        add_doc(doc, (1 - alpha) * b_score)

    # Step 6: Rank & Deduplicate
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

def get_sqlglot_dialect(database_engine):
    """Map database engine names to sqlglot dialect names."""
    dialect_map = {
        'postgresql': 'postgres',
        'postgres': 'postgres',
        'mysql': 'mysql',
        'mariadb': 'mysql',
        'sqlserver': 'tsql',
        'mssql': 'tsql',
        'tsql': 'tsql',
        'oracle': 'oracle',
        'snowflake': 'snowflake',
        'bigquery': 'bigquery',
        'sqlite': 'sqlite',
        'redshift': 'redshift',
        'hive': 'hive',
        'presto': 'presto',
        'spark': 'spark',
        'clickhouse': 'clickhouse',
        'databricks': 'databricks',
        'trino': 'trino',
        'duckdb': 'duckdb'
    }
    return dialect_map.get(database_engine.lower(), 'postgres')



def evaluate_query_rag(project_id, user_query, sql_gen, docs, database_engine="postgresql"):
    import sqlglot
    import re

    # Ensure docs is a list
    if not isinstance(docs, list):
        docs = [docs]

    # Normalize all docs to dicts with 'content'
    normalized_docs = []
    for d in docs:
        if hasattr(d, "page_content"):
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
    
    def sanitize_sql_block(sql_text: str) -> str:
        """
        Remove unsafe lines that break sqlglot parsing: SQL comments (-- ...), Oracle-style "/" delimiters, Placeholder ellipses, Empty lines
        """
        cleaned_lines = []
        for line in sql_text.splitlines():
            s = line.strip()

            # Skip comments
            if s.startswith("--"):
                continue
            
            # Skip Oracle delimiters or placeholders
            if s == "/" or "..." in s:
                continue
            
            # Skip empty lines
            if not s:
                continue

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    
    # --- Extract SQL from sql blocks ---
    def extract_sql_blocks(text):
        pattern = r"```sql(.*?)```"
        blocks = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if not blocks:
            blocks = re.findall(r"(SELECT .*?;)", text, flags=re.DOTALL | re.IGNORECASE)
        return [sanitize_sql_block(b) for b in blocks if sanitize_sql_block(b).strip()]

    # --- Extract SQL claims ---
    def extract_sql_claims(sql_text):
        import re
        import sqlglot
        from sqlglot.expressions import Table, Column, Join

        claims = []
        trees = []

        statements = [stmt.strip() for stmt in sql_text.split(";") if stmt.strip()]

        for stmt in statements:
            try:
                dialect = get_sqlglot_dialect(database_engine)
                parsed_trees = sqlglot.parse(stmt, dialect=dialect)  # Use appropriate dialect
                
                if not parsed_trees:
                    claims.append(f"parse_error:{database_engine}:No valid SQL parsed")
                    continue
            except Exception as e:
                # ✅ SYNTAX ERROR
                claims.append(f"parse_error:{database_engine}:{e}")
                continue

            for tree in parsed_trees:
                if not tree:
                    continue
                trees.append(tree)

                for t in tree.find_all(Table):
                    if t.name:
                        claims.append(f"table:{t.name.lower()}")

                for c in tree.find_all(Column):
                    if c.name:
                        col_name = c.name.lower().split(".")[-1]
                        claims.append(f"column:{col_name}")

                for j in tree.find_all(Join):
                    try:
                        on_expr = j.on() if callable(j.on) else j.on
                        if on_expr:
                            join_clause = re.sub(r"\s+", " ", on_expr.sql().lower()).strip()
                            claims.append(f"join:{join_clause}")
                    except Exception:
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


    # --- Debug ---
    print("Top 3 chunk texts:", [d["content"][:100] for d in normalized_docs[:3]])
    print("Extracted SQL claims:", all_claims)

    if all_claims and all_trees:
        precision1 = precision_at_k(normalized_docs, all_claims, 1)
        precision3 = precision_at_k(normalized_docs, all_claims, 3)
        precision5 = precision_at_k(normalized_docs, all_claims, 5)
    else:
        precision1 = precision3 = precision5 = None

    return {
        "Precision@1": precision1,
        "Precision@3": precision3,
        "Precision@5": precision5,
        "NumClaims": len(all_claims),
        "claims": all_claims
    }
    
# ----------------------------
# State Definition for LangGraph
# ----------------------------
class AgentState(TypedDict, total=False):
    project_id: int
    user_query: str
    conversation_id: str
    user_id: str
    memory: List[Dict[str, str]]
    database_engine: str
    query_intent: str
    retrieval_mode: str
    required_documents: List[str]
    next_steps: List[str]
    complexity: str
    keyword_search_words: List[str]
    contexts: List[Dict]
    generated_response: str
    attempt_count: int
    is_fallback: bool
    current_step: str
    response_type: str  
    generation_prompt: str
    rag_metrics: Dict[str, Any]
    validation_result: str
    needs_improvement: bool
    retry_target: str  # "retrieval", "generation", or "none"
    improvement_suggestions: List[str]
    reflection_count: int
    stream_chunks: Optional[Any]
    current_status: str

# ----------------------------
# Agent Functions
# ----------------------------
def planning_agent(state: AgentState) -> AgentState:
    """Analyze user query and plan the approach."""
    print("🔍 Planning Agent: Analyzing query intent...")
    prompt = f"""
    Analyze the user query and provide structured reasoning about how to handle it.

    USER QUERY: {state['user_query']}
    CONVERSATION HISTORY: {state['memory'][-20:] if state['memory'] else 'No history'}

    Analyze and return ONLY a JSON with these exact fields:
    - query_intent: "conversational", "sql_query", "sql_optimization"
    
    CRITICAL GUIDELINES:
    - If the query asks for data, information, analysis, reports, or anything that would require looking up data → "sql_query"
    - If the query mentions tables, columns, databases, schemas, SQL, queries, data, records → "sql_query"  
    - If the query asks "show me", "list", "find", "get", "how many", "what is the" followed by data-related terms which need retrieval → "sql_query"
    - If the user mentions terms like: "optimize", "tune", "performance", "slow query", "improve query", "make it faster", "explain plan", "explain analyze" AND includes or implies an existing SQL query → use "sql_optimization"
    - Only use "conversational" for greetings, thanks, casual chat, or questions about the assistant itself which does not need retrieval.

    For greetings/conversational (like "hello", "thanks", "how are you"), use:
    {{
        "query_intent": "conversational",
        "next_steps": ["generation"],
    }}
    For SQL Query (Like "Generate query to validate sales data"), use:
    {{
        "query_intent": "sql_query",
        "next_steps": ["retrieval"],
    }}
    For SQL Query Optimization (Like "Optimize this query"), use:
    {{
        "query_intent": "sql_optimization",
        "next_steps": ["optimization"],
    }}

    Return ONLY valid JSON, no other text.
    """
    
    try:
        # Use non-streaming call for planning
        result = call_llm(prompt)
        
        # Handle the return format properly
        if isinstance(result, tuple) and len(result) == 3:
            response_text = result[0]  # Extract the text from tuple (text, attempt_count, is_fallback)
        else:
            # Fallback if call_gemini returns something unexpected
            response_text = str(result) if result else '{"query_intent": "unknown", "next_steps": ["retrieval"]}'
        
        # Clean the response - remove any markdown formatting
        response_text = response_text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        try:
            planning_data = json.loads(response_text)
        except json.JSONDecodeError:
            # Extract JSON from malformed response or use fallback
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                planning_data = json.loads(json_match.group())
            else:
                planning_data = {"query_intent": "unknown"}
        
        state.update({
            "query_intent": planning_data.get("query_intent", "unknown"),
            "next_steps": planning_data.get("next_steps", []),
            "current_step": "planning_complete",
        })
        
        print(f"📋 Planning Result: {state['query_intent']}, Steps: {state['next_steps']}")
        
    except Exception as e:
        print(f"❌ Planning agent error: {e}")
        print(f"Raw response was: {response_text if 'response_text' in locals() else 'No response'}")
        
        # Default fallback based on query content
        user_query_lower = state['user_query'].lower()
        conversational_keywords = ['hello', 'hi', 'hey', 'thanks', 'thank you', 'bye', 'goodbye', 'how are you']
        optimization_keywords = ["optimize", "optimise", "tune", "tuning", "slow", "performance", "make faster", "improve query", "explain plan", "explain analyze"]
        
        if any(keyword in user_query_lower for keyword in conversational_keywords):
            state.update({
                "query_intent": "conversational",
                "next_steps": ["generation"],
                "current_step": "planning_complete",
            })
        elif any(k in user_query_lower for k in optimization_keywords):
            state.update({
                "query_intent": "sql_optimization",
                "next_steps": ["optimization"],
                "current_step": "planning_complete",
            })
        else:
            state.update({
                "query_intent": "sql_query", 
                "next_steps": ["retrieval"],
                "current_step": "planning_complete",
            })
        
        print(f"🔄 Using fallback planning: {state['query_intent']}")
    
    return state

# ----------------------------
# Retrieval Agent
# ----------------------------
def retrieval_agent(state: AgentState) -> AgentState:
    """Use LLM to analyze retrieval needs and fetch relevant documents."""
    print("🔍 Retrieval Agent: Analyzing retrieval needs...")
    
    # 🧩 Feedback-aware logic
    validation_feedback = state.get("rag_metrics", {}).get("LLMReasoning", "")
    validation_suggestions = state.get("improvement_suggestions", [])
    retry_target = state.get("retry_target", "none")
    reflection_count = state.get("reflection_count", 0)

    # Include validation insights if this is a retry after validation
    feedback_section = ""
    if retry_target == "retrieval" and (validation_feedback or validation_suggestions):
        feedback_section = f"""
        The previous validation step suggested retrieval improvement.
        REASONING: {validation_feedback}
        SUGGESTIONS: {', '.join(validation_suggestions) if validation_suggestions else 'None'}
        
        Based on this feedback, refine your retrieval analysis to better target the relevant information.
        """
        print("🧠 Applying validation feedback to refine retrieval...")
    
    
    prompt = f"""
    Analyze the user query for document retrieval needs. Return ONLY JSON with these exact fields:
    - retrieval_mode: "document_level" if user requests SQL or analysis for entire document, otherwise "chunk_level"
    - required_documents: list of specific document names mentioned or implied
    - keyword_search_words: list of key search terms for semantic search  
    - complexity: "low", "medium", or "high" based on query complexity

    USER QUERY: {state['user_query']}
    
    {feedback_section}

    Guidelines:
    
🔹 RETRIEVAL_MODE:
- "document_level" → when the query explicitly mentions full document processing or any document name mentioned with extension
   e.g., "generate SQL for whole document X", "analyze complete file Y", "use entire data model".
- "chunk_level" → when query targets specific info, e.g., "find Q4 sales", "get revenue of last year".

🔹 REQUIRED_DOCUMENTS:
- Extract ONLY explicit document names directly mentioned like pdf,doc,docx,csv,txt,xlsx file names.
- Must match exact document names from the database
- If no specific documents mentioned → empty list []
- Examples: 
  "Generate sql from document.csv" → ["document.csv"]
  "Analyze the document sample.docx" → ["sample.docx"]

🔹 KEYWORD_SEARCH_WORDS:
- Extract all MOST SPECIFIC nouns/entities from the query
- Focus on concrete business entities, not generic terms
- Skip verbs, adjectives, and generic words like "data", "information", "details"
- Examples:
  "info about colleges" → ["colleges"]
  "sales report for Q4" → ["sales", "Q4"]
  "customer orders last month" → ["customers", "orders"]
  "analyze revenue trends" → ["revenue", "trends"]

 🔹Complexity: "low" for simple lookups, "medium" for joins, "high" for complex analytics

    Return ONLY valid JSON, no other text.
    """
    
    try:
        actual_doc_names = get_project_documents(state['project_id'])
        if not actual_doc_names:
            print("⚠️ No documents found in this project. Skipping retrieval.")
            
            state.update({
                "contexts": [],
                "retrieval_mode": "none",
                "required_documents": [],
                "keyword_search_words": [],
                "complexity": "none",
                "retrieval_metrics": {
                    "documents_found": 0,
                    "keywords_used": 0,
                    "chunks_retrieved": 0,
                    "complexity_level": "none"
                },
                "current_step": "retrieval_complete",
                "retry_target": None
            })
            
            return state
        # Get LLM analysis for retrieval parameters
        result = call_llm(prompt)
        
        # Handle response format
        response_text = ""
        if isinstance(result, tuple) and len(result) == 3:
            response_text = result[0]
        elif isinstance(result, str):
            response_text = result
        else:
            response_text = str(result) if result else '{"required_documents": [], "keyword_search_words": [], "complexity": "medium"}'
            
        if "⚠️ All Gemini models are busy" in response_text or "❌ Something went wrong" in response_text:
            print("⚠️ LLM returned error message, using default retrieval parameters")
            retrieval_plan = {
                "retrieval_mode": "chunk_level",
                "required_documents": [],
                "keyword_search_words": [],
                "complexity": "medium"
            }
        else:
            # Clean the response
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            try:
                retrieval_plan = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse LLM response as JSON: {e}")
                print(f"Raw response: {response_text[:200]}...")
                # Use default values if JSON parsing fails
                retrieval_plan = {
                    "retrieval_mode": "chunk_level",
                    "required_documents": [],
                    "keyword_search_words": [],
                    "complexity": "medium"
                }
        
        # Extract the LLM reasoning results
        retrieval_mode = retrieval_plan.get("retrieval_mode","chunk_level")
        required_documents = retrieval_plan.get("required_documents", [])
        keyword_search_words = retrieval_plan.get("keyword_search_words", [])
        complexity = retrieval_plan.get("complexity", "medium")
        
        actual_doc_names = extract_doc_names_from_query(state['project_id'], state['user_query'])
        required_documents = [doc for doc in required_documents if doc in actual_doc_names]
        
        print(f"📋 Retrieval Plan:  Mode={retrieval_mode}, Docs={required_documents}, Keywords={keyword_search_words}, Complexity={complexity}")
        
        # Use existing logic to retrieve documents based on LLM analysis
        contexts = []
        
        # 1. Retrieve the whole document if user mentioned specific documents for the entire content
        if retrieval_mode == "document_level" and required_documents:
            print(f"📄 Full-document retrieval mode selected for: {required_documents}")
            for doc_name in required_documents:
                doc_chunks = retrieve_from_doc_by_name(state['project_id'], doc_name)
                contexts.extend(doc_chunks)
        
        
        # 2. Adjust retrieval parameters based on complexity
        if complexity == "high":
            threshold = 0.6  # Stricter similarity for complex queries
            max_candidates = 100  # More candidates for complex queries
        elif complexity == "low":
            threshold = 0.8  # Higher threshold for simple queries
            max_candidates = 20   # Fewer candidates for simple queries
        else:  # medium
            threshold = 0.7
            max_candidates = 50
        
        # 3. Retrieve project-wide relevant chunks using existing logic
        project_chunks = retrieve_relevant_chunks(
            state['project_id'], 
            state['user_query'],
            keyword_terms=keyword_search_words, 
            threshold=threshold, 
            max_candidates=max_candidates
        )
        contexts.extend(project_chunks)
        
        if not contexts:
            print("⚠️ Retrieval completed but no matching chunks found.")
            state.update({
                "contexts": [],
                "retrieval_mode": retrieval_mode,
                "required_documents": required_documents,
                "keyword_search_words": keyword_search_words,
                "complexity": complexity,
                "retrieval_metrics": {
                    "documents_found": len(required_documents),
                    "keywords_used": len(keyword_search_words),
                    "chunks_retrieved": 0,
                    "complexity_level": complexity
                },
                "current_step": "retrieval_complete"
            })
            return state
        
        # Remove duplicates using existing logic
        seen_texts = set()
        unique_contexts = []
        for doc in contexts:
            if doc.page_content not in seen_texts:
                unique_contexts.append(doc)
                seen_texts.add(doc.page_content)
        
        # Convert to context format
        context_json = []
        for chunk in unique_contexts:
            context_json.append({
                "content": chunk.page_content,
                "source": chunk.metadata.get("filename", "unknown"),
                "section": chunk.metadata.get("hierarchy", ""),
                "chunk_index": chunk.metadata.get("chunk_index", 0)
            })
        
        # Update state with retrieval results
        state.update({
            "retrieval_mode": retrieval_mode,
            "required_documents": required_documents,
            "keyword_search_words": keyword_search_words, 
            "complexity": complexity,
            "contexts": context_json,
            "retrieval_metrics": {
                "documents_found": len(required_documents),
                "keywords_used": len(keyword_search_words),
                "chunks_retrieved": len(context_json),
                "complexity_level": complexity
            },
            "current_step": "retrieval_complete"
        })
        
        print(f"✅ Retrieval complete: {len(context_json)} chunks retrieved")
        
    except Exception as e:
        print(f"❌ Retrieval agent error: {e}")
        # Fallback to basic retrieval without LLM analysis
        print("🔄 Using fallback retrieval...")
        
        # Use existing basic retrieval as fallback
        project_chunks = retrieve_relevant_chunks(state['project_id'], state['user_query'], keyword_terms=[])
        
        # Convert to context format
        context_json = []
        for chunk in project_chunks:
            context_json.append({
                "content": chunk.page_content,
                "source": chunk.metadata.get("filename", "unknown"),
                "section": chunk.metadata.get("hierarchy", ""),
                "chunk_index": chunk.metadata.get("chunk_index", 0)
            })
            
        print(f"✅ Retrieval complete: {len(context_json)} chunks retrieved")
        
        state.update({
            "retrieval_mode": "chunk_level",
            "required_documents": [],
            "keyword_search_words": [],
            "complexity": "medium", 
            "contexts": context_json,
            "current_step": "retrieval_complete"
        })
    
    return state

# ----------------------------
# Generation Agent
# ----------------------------

def generation_agent(state: AgentState) -> AgentState:
    """Generate response based on query intent and available contexts."""
    print("🤖 Generation Agent: Generating response...")
    
    try:
        database_engine = state.get("database_engine", "postgresql")
        validation_feedback = state.get("rag_metrics", {}).get("LLMReasoning", "")
        validation_suggestions = state.get("improvement_suggestions", [])
        retry_target = state.get("retry_target", "none")
        reflection_count = state.get("reflection_count", 0)
        
        retrieval_mode = state.get("retrieval_mode", "unknown")
        retrieval_metrics = state.get("retrieval_metrics", {})
        documents_found = retrieval_metrics.get("documents_found", 0)
        no_docs_present = (retrieval_mode == "none" or documents_found == 0)

        # Include validation insights if this is a retry after validation
        feedback_section = ""
        if retry_target == "generation" and (validation_feedback or validation_suggestions):
            feedback_section = f"""
            The previous validation step suggested generation improvement.
            REASONING: {validation_feedback}
            SUGGESTIONS: {', '.join(validation_suggestions) if validation_suggestions else 'None'}
            
            Based on this feedback, refine your generation.
            """
            print("🧠 Applying validation feedback to refine generation...")
                
                
        if state.get("query_intent") == "conversational":
            # Handle conversational queries (non-streaming for simplicity)
            print("💬 Generating conversational response...")
            conversational_prompt = f"""
            You are a friendly AI assistant. Respond naturally to the user's message.
            Always stay within the conversation topic or SQL/data-related context.
            Never fabricate information, speculate, or respond to out-of-scope topics (e.g., personal opinions, politics, weather, jokes).
            If the user asks something out of scope, politely decline and redirect back to SQL or data-related help.
            
            USER MESSAGE: {state['user_query']}
            CONVERSATION HISTORY: {state['memory'][-20:] if state['memory'] else 'No recent history'}
            
            {feedback_section}
            Respond in a friendly, helpful manner. Keep it concise and natural.
            """
            
            result = call_llm(conversational_prompt)
            if isinstance(result, tuple) and len(result) == 3:
                response_text = result[0]
            else:
                response_text = str(result) if result else "I'm here to help! How can I assist you today?"
            
            state.update({
                "generated_response": response_text,
                "response_type": "conversational",
                "current_step": "generation_complete"
            })
            
        elif state.get("query_intent") == "sql_query":
            # Handle SQL queries with contexts - prepare for streaming
            print("📊 Generating SQL response with contexts...")
            print("Current Database Engine:", database_engine)
            
            # Prepare contexts for the prompt
            contexts = state.get("contexts", [])
            final_json = json.dumps(contexts, separators=(",", ":")) if contexts else "[]"
            
            # Use your existing structured prompt
            structured_prompt = f"""
            # ROLE: Expert SQL Assistant
            # DATABASE ENGINE: {database_engine.upper()}
            # CONTEXT: You are analyzing documentation to generate precise SQL queries.
            ---
            
            {feedback_section}
            
            CONVERSATION HISTORY: {state['memory'][-20:] if state['memory'] else 'No recent history'}

            # AVAILABLE DOCUMENTATION:
            {final_json}

            ---

            # USER QUERY:
            "{state['user_query']}"
            
            # PROJECT DOCUMENT STATUS:
            {"No documents are available for this project." if no_docs_present else "Some documents were available for retrieval."}

            ---

            # INSTRUCTIONS:
            
            You MUST follow the reasoning process described in Sections 1,2,3,4,6 and think internally but your final output response should ONLY contain the formatted output described in Section 5 — nothing else.

            ## 1. QUERY ANALYSIS & INTENT
            - Analyze the user's question to understand what data is being requested. 
               -- **SQL_QUERY** → The user is asking for a data query, report, metric, aggregation, or data relationship. 
               -- **GENERAL_QUESTION** → The user is asking for an explanation, concept, or descriptive answer — not a SQL request.
               -- **OUT_OF_SCOPE** → The question is unrelated to data, SQL, or the provided documentation (e.g., weather, jokes, small talk). 
            - Identify the key **entities**, **metrics**, **filters**, and **relationships** involved.  
            - Determine whether this is a **simple lookup**, **aggregation**, **join**, or **complex analytical query**.
            
            ---
            
            ## 2. SCHEMA UNDERSTANDING (From Documentation)
            **CRITICAL:** Carefully extract and verify the following information from the provided context:
            - Tables available and their purposes.  
            - Columns with exact names and data types.  
            - Primary Keys (`PK:`) and Foreign Keys (`FK:`).  
            - Relationships between tables.  
            - Business rules or constraints mentioned in the documentation.  
            - If the context is insufficient, apply standard SQL design patterns reasonably.
            
            ### CROSS-REFERENCE TRAVERSAL RULES
            - If Step 10 in retrieved contexts says "go to Step 3", search the contexts for Step 3 and apply that information from the chunks.
            - If Document A references Document B, use the relevant B content already present in the retrieved contexts.
            - If a referenced step or document cannot be found in the contexts, explicitly state: "Referenced section not available in retrieved context."
            - Combine and interpret all referenced sections accurately and avoid hallucinating missing information.

            ---

            ## 3. SQL GENERATION REQUIREMENTS
            -- **DATABASE ENGINE**: {database_engine.upper()}
            -- Generate a syntactically correct SQL query using {database_engine.upper()}-specific syntax.
            -- **CRITICAL**: Research and apply {database_engine.upper()}-specific features and optimizations.
            -- Use exact table and column names from the documentation.
            -- Include proper JOIN conditions using PK/FK relationships.
            -- Apply relevant WHERE clauses, aggregations, and ordering.
            -- **CRITICAL:** Always generate a performance-optimized and cost-efficient SQL query using {database_engine.upper()}
            
            -- STRICT DIALECT MODE (CRITICAL):
            -- Write SQL as if this query must pass a strict production linting review for {database_engine.upper()}.
            -- Prefer {database_engine.upper()}’s native functions over generic ANSI SQL.
            -- Replace any cross-dialect constructs with the idiomatic dialect equivalents.
            -- Perform an internal dialect compliance check before outputting SQL.
            -- Do NOT write “generic SQL”. Write REAL SQL dialect specific for {database_engine.upper()}.

            -- Formatting Guidelines:
            --   • Use clear indentation and line breaks.
            --   • Any SQL inside ```sql``` blocks MUST be syntactically complete, executable SQL for {database_engine.upper()}.
            --   • Do NOT include pseudo-code, comments describing omitted sections, ellipses (...), or Oracle-style "/" delimiters inside SQL blocks.
            --   • DO NOT include: SQL comments (-- or /* */), Explanatory text, Section headers, Inline notes within SQL blocks.
            --   • If explanation is required, it MUST appear strictly AFTER the SQL block.
            --   • If something cannot be written as valid executable SQL, describe it ONLY outside code blocks.
            --   • Ensure the query directly answers the user's question.
            
            ---

            ## 4. EXPLANATION & SCHEMA DIAGRAM
            - Generate an ER diagram-like ASCII structure in separate copy-friendly format:
            - Each table in its own box (┌─┐ │ └─┘)
            - PKs with "PK:" prefix, FKs with "[FK]"
            - Show joins with ◄────► arrows
            - Place aggregation/filter/GROUP BY/HAVING/ORDER BY/other conditions boxes separately and join them.
            - Keep it compact and aligned, like a schema diagram.
            
            ---

            ## 5. FINAL OUTPUT FORMAT — ADAPTIVE RESPONSE GENERATION
            
            ### FORMATTING REQUIREMENTS:
            - Use CLEAR SECTION BREAKS with blank lines between major sections
            - Separate different points with line breaks for better readability
            - Use bullet points with proper spacing between each item
            - Add blank lines before and after code blocks
            - Ensure paragraphs have proper spacing for easy reading
            - If multiple SQL queries are needed, repeat the TEMPLATE for each query.
            - DO NOT combine multiple queries into a single SQL block.
            - Do NOT repeat headings, context, or restate the user request.
            
            ### CASE 1: If intent == "SQL_QUERY"
            Follow this full, structured output:
            Provide a brief natural-language summary that directly answers the user's question.

            **SQL Query**
            SQL queries should be in copy-friendly format within "```sql ```".
            ```sql SELECT * FROM ... <Without comments or explanations>```

            **Explanation Diagram**
            ASCII diagram showing ER relationship diagrams with tables, relationships, and key operations "```  Diagram ```".

            **Notes**
            * List any assumptions made.
            * Mention limitations or potential ambiguities.
            * Suggest further analysis if applicable.
            
            ### CASE 2: If intent == "GENERAL_QUESTION"

            * Do NOT include SQL, diagrams, or “Notes” placeholders.
            * Provide a concise and factual explanation directly answering the question.
            * If relevant, refer to documentation or general SQL best practices.
            * If some info is missing, politely state what’s unavailable and suggest next steps.
            
            ### CASE 3: If intent == "OUT_OF_SCOPE"

            * Acknowledge politely that the question is outside SQL or documentation scope.
            * Do NOT generate any SQL or technical placeholders.
            * Respond conversationally, in a friendly and helpful tone.
            * Optionally offer to switch back to SQL/data assistance.
            
            ---

            ## 6. CONVERSATIONAL HANDLING

            - If the input is a greeting, thank-you, or general conversation:
            - Respond naturally and politely.
            - Maintain a professional yet friendly tone.
            - Optionally offer SQL or data assistance if relevant.
            - Keep responses concise and engaging.
            
            🚨 HARD OUTPUT CONSTRAINT (NON-NEGOTIABLE)

            If ANY of the following occur, the response is INVALID:
            - SQL code blocks contain comments (-- or /* */)
            - SQL blocks contain explanatory text
            - The same SQL statement is repeated
            - Narrative text appears before the first SQL block

            If invalid, you MUST regenerate silently and output only the corrected version.

            """
            
            ## Stream the response and collect chunks
            print("🔄 Streaming response from LLM...")
            stream_chunks = []
            full_response = ""
            attempt_count = 0
            is_fallback = False
            response_text, attempt_count, is_fallback = call_gemini(
                structured_prompt,
                stream=False
            )
            full_response += response_text
            stream_chunks.append(response_text)
            state["attempt_count"] = attempt_count
            state["is_fallback"] = is_fallback
            state["generated_response"] = full_response
            
            print(f"✅ Collected {len(stream_chunks)} chunks, total length: {len(full_response)}")
            
            # Store everything in state
            state.update({
                "generation_prompt": structured_prompt,
                "generated_response": full_response,
                "stream_chunks": stream_chunks,
                "response_type": "sql",
                "contexts_used": contexts,
                "current_step": "generation_complete"
            })
        
        print(f"✅ Generation complete. Type: {state.get('response_type', 'unknown')}")
        return state
        
        
    except Exception as e:
        print(f"❌ Generation agent error: {e}")
        # Fallback response
        state.update({
            "generated_response": "I apologize, but I encountered an error while generating the response. Please try again.",
            "response_type": "error",
            "current_step": "generation_complete"
        })
    
    return state


# ----------------------------
# Validation Agent
# ----------------------------
def validation_agent(state: AgentState) -> AgentState:
    """Evaluate the RAG response quality and provide improvement suggestions."""
    print("🔍 Validation Agent: Evaluating response quality...")
    
    try:
        # Only validate SQL responses that have been generated
        if state.get("response_type") in ["sql", "sql_optimization"]:
            retrieval_mode = state.get("retrieval_mode", "unknown")
            retrieval_metrics = state.get("retrieval_metrics", {})
            documents_found = retrieval_metrics.get("documents_found", 0)
            no_docs_present = (retrieval_mode == "none" or documents_found == 0)
            generated_response = state["generated_response"]
            contexts = state.get("contexts", [])
            user_query = state["user_query"]
            project_id = state["project_id"]
            
            # Get database_engine from state
            database_engine = state.get("database_engine", "postgresql")
            
            # Run basic RAG evaluation (only reliable rule-based metrics)
            rag_metrics = evaluate_query_rag(project_id, user_query, generated_response, contexts, database_engine=database_engine)

            # ==========================================================
            # FORCE RETRY ON SQL SYNTAX / SEMANTIC ERRORS
            # ==========================================================
            
            claims = rag_metrics.get("claims", [])

            sql_errors = [
                c for c in claims
                if c.startswith("parse_error") or c.startswith("semantic_error")
            ]

            if sql_errors:
                print("🚨 SQL validation failure detected:", sql_errors)
                # Get current reflection count
                current_count = state.get("reflection_count", 0)
                
                if current_count < 2:
                    # Increment reflection count
                    new_count = current_count + 1
                    state["reflection_count"] = new_count
                    
                    state.update({
                        "rag_metrics": rag_metrics,
                        "validation_result": f"SQL validation failed (attempt {new_count}/2): " + "; ".join(sql_errors),
                        "needs_improvement": True,
                        "retry_target": ("optimization" if state.get("query_intent") == "sql_optimization" else "generation"),
                        "improvement_suggestions": sql_errors,
                        "current_step": "validation_complete"
                    })
                    
                    print(f"🔄 Will retry (attempt {new_count}/2)")
                else:
                    # Max retries reached - force stop
                    state.update({
                        "rag_metrics": rag_metrics,
                        "validation_result": f"❌ Max retries ({current_count}) reached. SQL validation failed: " + "; ".join(sql_errors),
                        "needs_improvement": False,  # IMPORTANT: Force stop
                        "retry_target": "none",      # IMPORTANT: Force stop
                        "improvement_suggestions": ["Max retry attempts reached. Please check your SQL syntax manually."],
                        "current_step": "validation_complete"
                    })
                    print(f"⏹️ Max retries ({current_count}) reached, stopping loop")

                return state
            
            # Add LLM reasoning for problematic metrics
            llm_eval_prompt = f"""
            As a SQL and data analysis expert, evaluate this RAG system output.
            
            DATABASE ENGINE: {database_engine}

            USER QUERY: "{user_query}"
            
            GENERATED RESPONSE:
            {generated_response}
            
            AVAILABLE CONTEXTS:
            {(contexts)}
            
            PROJECT DOCUMENT STATUS:
            {"No documents are available for this project. Retrieval was skipped." if no_docs_present else "Some documents were available for retrieval."}
            
            RETRIEVAL METRICS (Rule-based):
            - Precision@1: {rag_metrics.get('Precision@1', 'N/A')}
            - Precision@3: {rag_metrics.get('Precision@3', 'N/A')}
            - Precision@5: {rag_metrics.get('Precision@5', 'N/A')}
            - SQL Elements Found: {rag_metrics.get('NumClaims', 'N/A')}
            
            Evaluate these specific aspects:

            1. QUERY COVERAGE (0.0-1.0):
            - Does the generated response fully address the user's original query intent?
            - Consider both explicit requirements and implicit needs
            - Score: 1.0 = perfectly addresses, 0.0 = completely misses

            2. CLAIM SUPPORT RATE (0.0-1.0):
            - What percentage of SQL elements (tables, columns, joins, conditions) are supported by the available contexts?
            - Consider semantic support, not just exact string matches
            - Include synonyms and related terms
            - Score: 1.0 = fully supported, 0.0 = completely unsupported

            3. STATIC EXECUTION SCORE (0.0-1.0):
            - How correct and executable is the generated SQL?
            - Consider syntax, logic, and potential execution success
            - Score: 1.0 = perfectly executable, 0.0 = completely invalid
            
            4. RETRY DECISION ANALYSIS:
            - Analyze if the main issue is RETRIEVAL (bad contexts) or GENERATION (bad SQL)
            - RETRIEVAL issues: contexts missing key info, low precision scores
            - GENERATION issues: good contexts but poor SQL, syntax errors, or outputs that are disorganized, messy, poorly structured, or contain duplicated content that needs to be regenerated.
            - OPTIMIZATION issues: SQL is valid but inefficient, unoptimized, or slow and query intent is sql_optimization
            - If both bad but retrieval worse → RETRIEVAL
            - Default to RETRIEVAL if unclear
            - If NO DOCUMENTS are available for the Project, DO NOT suggest retrieval retry.

            Return ONLY JSON with these exact fields:
            - "QueryCoverage": number between 0.0-1.0
            - "ClaimSupportRate": number between 0.0-1.0  
            - "StaticExecScore": number between 0.0-1.0
            - "needs_improvement": true or false
            - "retry_target": "retrieval" or "generation" or "optimization" or "none"
            - "improvement_suggestions": ["list", "of", "specific", "suggestions"]
            - "reasoning": "Detailed analysis of scores and retry decision"

            Return ONLY valid JSON, no other text.
            
            5. 🚨 HARD OUTPUT CONSTRAINT (NON-NEGOTIABLE)

            If ANY of the following occur, the response is INVALID:
            - SQL code blocks contain comments (-- or /* */)
            - SQL blocks contain explanatory text
            - The same SQL statement is repeated
            - Narrative text appears before the first SQL block

            If invalid, you MUST regenerate silently and output only the corrected version.

            """
            
            try:
                result = call_llm(llm_eval_prompt)
                response_text = result[0] if isinstance(result, tuple) else str(result)
                
                # Clean response
                response_text = response_text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                llm_analysis = json.loads(response_text)
                
                # Add LLM metrics to rag_metrics with same names
                rag_metrics.update({
                    "QueryCoverage": llm_analysis.get("QueryCoverage", 0.5),
                    "ClaimSupportRate": llm_analysis.get("ClaimSupportRate", 0.5),
                    "StaticExecScore": llm_analysis.get("StaticExecScore", 0.5),
                    "LLMReasoning": llm_analysis.get("reasoning", "No detailed reasoning"),
                    "needs_improvement": llm_analysis.get("needs_improvement", False),
                    "retry_target": llm_analysis.get("retry_target", "none"),
                    "improvement_suggestions": llm_analysis.get("improvement_suggestions", [])
                })

                print(f"🤖 LLM Analysis: Needs improvement = {rag_metrics['needs_improvement']}, Retry target = {rag_metrics['retry_target']}")
                print(f"🧠 LLM Reasoning: {rag_metrics['LLMReasoning']}, Suggestions: {rag_metrics['improvement_suggestions']}")
                
            except Exception as e:
                print(f"❌ LLM evaluation error: {e}")
                # Fallback scores for LLM metrics
                rag_metrics.update({
                    "QueryCoverage": None,
                    "ClaimSupportRate": None,
                    "StaticExecScore": None,
                    "LLMReasoning": f"LLM evaluation failed: {str(e)}",
                    "needs_improvement": False,  # Default to improvement needed on error
                    "retry_target": "none",  # Default to retrieval retry
                    "improvement_suggestions": ["Check system configuration and try again"]
                })
            
            # Generate validation summary
            validation_summary = f"""
            ## RAG Quality Assessment

            ### 📊 Metrics Summary:
            - **Precision@1**: {rag_metrics['Precision@1'] or 'N/A'} (Retrieval Quality)
            - **Precision@3**: {rag_metrics['Precision@3'] or 'N/A'} (Retrieval Quality) 
            - **Precision@5**: {rag_metrics['Precision@5'] or 'N/A'} (Retrieval Quality)
            - **Claim Support Rate**: {rag_metrics['ClaimSupportRate'] or 'N/A'} (LLM: Context Support)
            - **Static Execution Score**: {rag_metrics['StaticExecScore'] or 'N/A'} (LLM: SQL Quality)
            - **Query Coverage**: {rag_metrics['QueryCoverage'] or 'N/A'} (LLM: Intent Match)
            - **SQL Elements Found**: {rag_metrics['NumClaims']} (Rule-based)

            ### 🧠 LLM Analysis:
            {rag_metrics.get('LLMReasoning', 'No analysis available')}

            ### 🔄 Retry Decision:
            - **Needs Improvement**: {rag_metrics.get('needs_improvement', False)}
            - **Retry Target**: {rag_metrics.get('retry_target', 'none')}
            - **Suggestions**: {', '.join(rag_metrics.get('improvement_suggestions', ['No specific suggestions']))}

            ### 🎯 Status: {'⚠️ Needs Improvement' if rag_metrics.get('needs_improvement', False) else '✅ Quality Acceptable'}
            """
            current_count = state.get("reflection_count", 0)
            if rag_metrics.get("needs_improvement") and current_count < 2:
               current_count = current_count + 1
               state["reflection_count"]=current_count
               print(f"✅ Final reflection count: {current_count}")
            elif current_count >= 2:
                rag_metrics["needs_improvement"] = False
                rag_metrics["retry_target"] = None
                print(f"✅ Final reflection count: {current_count}")
            
            state.update({
                "rag_metrics": rag_metrics,
                "validation_result": validation_summary,
                "needs_improvement": rag_metrics.get('needs_improvement', False),
                "retry_target": rag_metrics.get('retry_target', 'none'),
                "improvement_suggestions": rag_metrics.get('improvement_suggestions', []),
                "current_step": "validation_complete"
            })
            
            if state["query_intent"] == "sql_optimization":
                if state["retry_target"] == "generation":
                    rag_metrics["retry_target"] = "optimization"
                    state["retry_target"] = "optimization"
            print(f"✅ Hybrid validation complete. Needs improvement: {rag_metrics.get('needs_improvement', False)}")
            print(f"🎯 Retry target: {rag_metrics.get('retry_target', 'none')}")
            
        else:
            state.update({
                "rag_metrics": {},
                "validation_result": "Validation skipped - not a SQL response",
                "needs_improvement": False,
                "retry_target": "none",
                "improvement_suggestions": [],
                "current_step": "validation_complete"
            })
            print("⏭️  Validation skipped (not a SQL response)")
            
    except Exception as e:
        print(f"❌ Validation agent error: {e}")
        current_count = state.get("reflection_count", 0) + 1
        state.update({
            "rag_metrics": {},
            "reflection_count": current_count,
            "validation_result": f"Validation failed: {str(e)}",
            "needs_improvement": current_count < 2,
            "retry_target": "retrieval" if current_count < 2 else "none",
            "improvement_suggestions": ["System error occurred - retry recommended"],
            "current_step": "validation_complete"
        })
    
    return state

# ----------------------------
# Optimization Agent
# ----------------------------
def optimization_agent(state: AgentState) -> AgentState:
    """
    Handles SQL optimization requests using LLM reasoning only.  
    """

    print("⚡ Optimization Agent: Processing SQL optimization request...")
    
    database_engine = state.get("database_engine", "postgresql")
    
    validation_feedback = state.get("rag_metrics", {}).get("LLMReasoning", "")
    validation_suggestions = state.get("improvement_suggestions", [])
    retry_target = state.get("retry_target", "none")
    reflection_count = state.get("reflection_count", 0)
    
    retrieval_mode = state.get("retrieval_mode", "unknown")
    retrieval_metrics = state.get("retrieval_metrics", {})
    documents_found = retrieval_metrics.get("documents_found", 0)
    no_docs_present = (retrieval_mode == "none" or documents_found == 0)

    # Include validation insights if this is a retry after validation
    feedback_section = ""
    if retry_target == "optimization" and (validation_feedback or validation_suggestions):
        feedback_section = f"""
        The previous validation step suggested optimization improvement.
        REASONING: {validation_feedback}
        SUGGESTIONS: {', '.join(validation_suggestions) if validation_suggestions else 'None'}
        
        Based on this feedback, refine your optimization.
        """
        print("🧠 Applying validation feedback to refine optimization...")

    prompt = f"""
You are a senior SQL Performance Engineer for an enterprise-grade optimization platform.

## YOUR UNIQUE CAPABILITY:
This system specializes in **schema-aware SQL optimization** by intelligently interpreting:
1. **Execution plans** (EXPLAIN / EXPLAIN ANALYZE)
2. **User-provided context**
3. **Available documentation**

You can automatically **infer database structures** and **detect performance patterns** that typically require hours of manual analysis.

## AUTOMATIC DETECTION CAPABILITIES:
From execution plans and minimal context, you can identify:
- **Full table scans** and missing index opportunities
- **Incorrect join ordering** and inefficient join methods
- **Late filtering** causing excessive row processing
- **Hash join spills** and memory pressure indicators
- **Missing partitioning** benefits
- **Suboptimal data type usage**
- **Statistics estimation errors**
- **Parallel execution inefficiencies**

---

{feedback_section}

## CONTEXTUAL INTELLIGENCE:
**CONVERSATION HISTORY**: {state['memory'][-20:] if state['memory'] else 'No recent history'}
**USER REQUEST**: "{state['user_query']}"
**AVAILABLE SCHEMA / DOCUMENTATION**: {json.dumps(state.get("contexts", []), indent=2)}
**PROJECT STATUS**: {"No documents available - you must infer schema from query patterns and execution plans." if no_docs_present else "Documentation available for schema-aware analysis."}

---

## INTELLIGENT ANALYSIS PROCESS:

### STEP 1: SCHEMA INFERENCE
- **Infer table structures** from query patterns and EXPLAIN plans
- **Estimate relationships** between tables based on join conditions
- **Identify potential indexes** from WHERE clauses and JOIN conditions
- **Detect partitioning opportunities** from date ranges and filtering patterns

### STEP 2: PLAN INTERPRETATION
- **Read between the lines** of EXPLAIN output to understand hidden issues
- **Spot data volume mismatches** between estimated and actual rows
- **Identify resource bottlenecks** (memory, I/O, CPU)
- **Detect sequential vs parallel execution inefficiencies**

### STEP 3: HOLISTIC OPTIMIZATION
- **Combine schema inference** with plan analysis for comprehensive optimization
- **Consider database engine specifics** for {database_engine.upper()}
- **Balance immediate fixes** with architectural improvements

---

## CRITICAL GUIDELINES:
- BE PROACTIVE: Don't wait for explicit problem statements - detect issues automatically
- BE SCHEMA-INTELLIGENT: Infer structure from what's available
- BE ENGINE-SPECIFIC: Use {database_engine.upper()}'s unique capabilities
- BE ACTIONABLE: Provide executable SQL, not just theory.
- BE CONCISE: Focus on high-impact optimizations
- BE PRACTICAL: Recommend changes that work in production environments
- Any SQL inside ```sql``` blocks MUST be syntactically complete, executable SQL for {database_engine.upper()}.
- Do NOT include pseudo-code, comments describing omitted sections, ellipses (...), or Oracle-style "/" delimiters inside SQL blocks.
- DO NOT include: SQL comments (-- or /* */), Explanatory text, Section headers, Inline notes within SQL blocks.
- If explanation is required, it MUST appear strictly AFTER the SQL block.
- If something cannot be written as valid executable SQL, describe it ONLY outside code blocks.

## OUTPUT FORMATTING:

### SCENARIO 1: SQL + EXPLAIN PLAN PROVIDED
**Perform deep analysis and provide:**
1. **Performance Diagnosis** - What you automatically detected
2. **Root Causes** - Why these issues exist
3. **Optimized SQL** - Specific improvements
4. **Schema Recommendations** - Indexes, partitions, statistics
5. **Expected Plan Improvement** - How execution changes

**Format for {database_engine.upper()}:**

**Performance Issues Detected**
[Bulleted list of what you automatically identified]

**Optimized SQL for {database_engine.upper()}**
```sql 
<Without comments or explanations>
<engine-optimized query>
```

**Recommended Database Improvements**
If needed, recommend specific schema changes otherwise Don't mention this section.
```sql
<Without comments or explanations>
<CREATE INDEX, ALTER TABLE, PARTITION, etc.>
```

**Expected Execution Plan (After Optimization)**
```
Explain Plan
```

### SCENARIO 2: SQL PROVIDED, NO EXPLAIN
Use schema inference and ask targeted questions:

"Based on the SQL structure, I can infer potential issues with [specific aspects]. To provide precise optimizations, please share:

1. EXPLAIN ANALYZE output for this query
2. Approximate table sizes
3. Existing indexes on tables: [list relevant tables]"

**Preliminary Recommendations (Based on SQL Patterns)**
```sql
<Without comments or explanations>
<Initial optimization suggestions>
```

### SCENARIO 3: NO SQL PROVIDED

"Please provide the SQL query you'd like optimized. For best results, include:

1. The complete SQL query
2. EXPLAIN or EXPLAIN ANALYZE output if available
3. Any known performance issues or context"

"""

    try:
        response = call_llm(prompt)

        if isinstance(response, tuple) and len(response) == 3:
            response_text = response[0]
        else:
            response_text = str(response)

        # Save response
        state.update({
            "generated_response": response_text.strip(),
            "response_type": "sql_optimization",
            "current_step": "optimization_complete",
        })

        print("✅ Optimization processing complete")

    except Exception as e:
        print("❌ Optimization agent failed:", e)
        state.update({
            "generated_response": (
                "I ran into an error while analyzing your query. "
                "Please try again or re-send the SQL you want optimized."
            ),
            "response_type": "error",
            "current_step": "optimization_complete",
        })

    return state


def update_status_before_node(state: AgentState, node_name: str):
    
    retry_target = state.get("retry_target")

    # Convert normal node into retry node if this is a retry
    if retry_target == node_name:
        node_name = f"retry_{node_name}"
    
    status_messages = {
        "planning": "🔍 Analyzing query...",
        "retrieval": "📋 Retrieving information...", 
        "generation": "🤖 Generating response...",
        "optimization": "⚡ Optimizing SQL...",
        "validation": "✅ Validating answer...",
        "retry_retrieval": "🔄 Improving retrieval...",
        "retry_generation": "🔄 Improving response...",
        "retry_optimization": "🔄 Reworking optimization..."
    }
    state["current_status"] = status_messages.get(node_name, "Processing...")
    return state

from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

# ----------------------------
# LangGraph Workflow Construction
# ----------------------------
def create_agent_workflow():
    """Create the LangGraph workflow with agentic decision making."""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("planning", lambda state: planning_agent(update_status_before_node(state, "planning")))
    workflow.add_node("retrieval", lambda state: retrieval_agent(update_status_before_node(state, "retrieval")))
    workflow.add_node("generation", lambda state: generation_agent(update_status_before_node(state, "generation")))
    workflow.add_node("validation", lambda state: validation_agent(update_status_before_node(state, "validation")))
    workflow.add_node("optimization", lambda state: optimization_agent(update_status_before_node(state, "optimization")))

    # Define workflow edges
    workflow.set_entry_point("planning")
    workflow.add_conditional_edges(
        "planning",
        lambda state:
            "generation" if state["query_intent"] == "conversational"
            else "retrieval",
        {
            "generation": "generation",
            "retrieval": "retrieval",
        }
    )
    workflow.add_conditional_edges(
        "retrieval",
        lambda state:
            "optimization"
            if state["query_intent"] == "sql_optimization"
            else "generation",
        {
            "optimization": "optimization",
            "generation": "generation"
        }
    )
    workflow.add_edge("optimization", "validation")
    workflow.add_edge("generation", "validation")
    workflow.add_conditional_edges(
        "generation",
        lambda state: "validation" if state.get("response_type")  in ["sql", "sql_optimization"] and state.get("current_step") == "generation_complete" else END,
        {
            "validation": "validation",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "validation",
        lambda state: (
            state.get("retry_target")
            if (
                state.get("needs_improvement", False)
                and state.get("retry_target") in ["retrieval", "optimization", "generation"]
                and state.get("reflection_count", 0) < 2
            )
            else END
        ),
        {
            "retrieval": "retrieval",
            "optimization": "optimization",
            "generation": "generation",
            END: END
        }
    )
    
    return workflow.compile(checkpointer=checkpointer)

# Create the workflow
agent_workflow = create_agent_workflow()


# ----------------------------
# Chat Model
# ----------------------------

class ChatRequest(BaseModel):
    project_id: int
    query: str
    conversation_id: Optional[str] = None  # optional
    top_k: int = 5
    original_query_id: Optional[str] = None  # << added
    database_engine: Optional[str] = None


@app.post("/chat")
async def chat(req: ChatRequest, current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    conversation_id = req.conversation_id or str(uuid.uuid4())
    safe_query = redact_pii(req.query)
    safe_query = sanitize_user_query(safe_query)
    start_time = time.time()


    # Store query in database initially — reuse original_query_id if provided
    query_id = req.original_query_id or str(uuid.uuid4())

    conn = get_connection()
    c = conn.cursor()

    if req.original_query_id:
        # If provided, try to update existing row; if missing, insert a new row with that id
        c.execute("SELECT id FROM queries WHERE id=?", (req.original_query_id,))
        if c.fetchone():
            c.execute("""
                UPDATE queries
                SET query = ?, answer = ?, response_time = ?, attempts = ?, is_fallback = ?, database_engine = ?
                WHERE id = ?
            """, (safe_query, "", 0, 0, False, req.database_engine, req.original_query_id))
        else:
            c.execute("""
                INSERT INTO queries (id, project_id, query, answer, user_id, conversation_id, response_time, database_engine)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (query_id, req.project_id, safe_query, "", user_id, conversation_id, 0, req.database_engine))
    else:
        # original not provided — insert new as before
        c.execute("""
            INSERT INTO queries (id, project_id, query, answer, user_id, conversation_id, response_time, database_engine)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (query_id, req.project_id, safe_query, "", user_id, conversation_id, 0, req.database_engine))

    conn.commit()
    conn.close()

    # Initialize memory if missing
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
    
    # ✅ ADD DEBUG: Show full conversation history
    print(f"🧠 FULL Conversation History for {conversation_id}: {len(memory)} messages")
    for i, msg in enumerate(memory):
        print(f"  {i}: {msg['role']}: {msg['content'][:80]}...")

    memory.append({"role": "user", "content": safe_query})
    
    # Initialize state for LangGraph
    initial_state = AgentState(
        user_query=safe_query,
        project_id=req.project_id,
        conversation_id=conversation_id,
        user_id=user_id,  
        database_engine=req.database_engine,
        memory=memory,
        query_intent="",
        next_steps=[],
        retrieval_mode="",
        required_documents=[],
        complexity="",
        keyword_search_words=[],
        generated_response="",
        contexts=[],
        attempt_count=0,
        is_fallback=False,
        current_step="start",
        rag_metrics={},
        validation_result="",
        needs_improvement=False,
        retry_target="none",
        improvement_suggestions=[],
        reflection_count=0
    )
    config = {
        "configurable": {
            "thread_id": conversation_id  # ✅ Only needed
        }
    }
    
        
    
    # Streaming response generator
    async def stream_and_save():
        # Send initial status
        yield "[[[STATUS]]]🚀 Starting processing...---SEPARATOR---"
        
        config = {
            "configurable": {
                "thread_id": conversation_id
            }
        }
        
        # Track previous status to avoid duplicates
        previous_status = ""
        
        # Stream status updates during workflow execution
        for current_state in agent_workflow.stream(initial_state, config=config):
            for node_name, state in current_state.items():
                if "current_status" in state and state["current_status"] != previous_status:
                    yield f"[[[STATUS]]]{state['current_status']}---SEPARATOR---"
                    await asyncio.sleep(0.01)
                    previous_status = state["current_status"]
                final_state = state
        
        workflow_state = final_state if final_state else agent_workflow.invoke(initial_state, config=config)
        
        # Get the final response
        full_answer = workflow_state.get("generated_response", "")
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Store assistant response in memory
        memory.append({"role": "assistant", "content": full_answer})
        
        # Truncate memory if needed
        if len(memory) > MAX_MEMORY_LENGTH:
            conversation_memory[conversation_id] = memory[-MAX_MEMORY_LENGTH:]
            
        rag_metrics = workflow_state.get("rag_metrics", {})
        
        # Update database with final answer
        conn = get_connection()
        c = conn.cursor()
        c.execute("""
            UPDATE queries
            SET answer = ?, response_time = ?, attempts = ?, is_fallback = ?, 
                precision1 = ?, precision3 = ?, precision5 = ?, 
                claim_support = ?, static_exec = ?, query_coverage = ?, num_claims = ?
            WHERE id = ?
        """, (full_answer,
            response_time,
            workflow_state.get("attempt_count", 1),
            workflow_state.get("is_fallback", False),
            rag_metrics.get("Precision@1"),
            rag_metrics.get("Precision@3"),
            rag_metrics.get("Precision@5"),
            rag_metrics.get("ClaimSupportRate"),
            rag_metrics.get("StaticExecScore"),
            rag_metrics.get("QueryCoverage"),
            rag_metrics.get("NumClaims"),
            query_id))
        conn.commit()
        conn.close()
        
        # Send the COMPLETE response at the end (not in chunks)
        yield f"{full_answer}---SEPARATOR---"
        
        meta = {
            "contexts": workflow_state.get("contexts", []),
            "query_id": query_id,
            "conversation_id": conversation_id
        }


        # Send metadata at the end
        yield f"[[[META]]]{json.dumps(meta)}---SEPARATOR---"

    return StreamingResponse(stream_and_save(), media_type="text/plain")
    

# ----------------------------
# Update Query Endpoint
# ----------------------------
class QueryUpdateRequest(BaseModel):
    query: Optional[str] = None
    answer: Optional[str] = None

@app.put("/queries/{query_id}")
async def update_query(
    query_id: str, 
    req: QueryUpdateRequest, 
    current_user: dict = Depends(get_current_user)
):
    """
    Update an existing query in the database.
    """
    conn = get_connection()
    c = conn.cursor()
    
    try:
        # Verify the query exists and belongs to the user
        c.execute("SELECT id FROM queries WHERE id=? AND user_id=?", (query_id, current_user["user_id"]))
        existing_query = c.fetchone()
        
        if not existing_query:
            conn.close()
            raise HTTPException(status_code=404, detail="Query not found")
        
        # Build update query dynamically based on provided fields
        update_fields = []
        update_values = []
        
        if req.query is not None:
            update_fields.append("query = ?")
            update_values.append(req.query)
        
        if req.answer is not None:
            update_fields.append("answer = ?")
            update_values.append(req.answer)
        
        if not update_fields:
            conn.close()
            raise HTTPException(status_code=400, detail="No fields to update")
        
        # Add query_id to values
        update_values.append(query_id)
        
        # Execute update
        update_query = f"UPDATE queries SET {', '.join(update_fields)} WHERE id = ?"
        c.execute(update_query, update_values)
        conn.commit()
        
        conn.close()
        return {"status": "success", "message": "Query updated successfully"}
        
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Failed to update query: {str(e)}")

# ----------------------------
# Regeneration Endpoint
# ----------------------------
class RegenerateRequest(BaseModel):
    project_id: int
    query: str
    query_id: str  # Existing query ID to update
    conversation_id: Optional[str] = None
    database_engine: Optional[str] = None

@app.post("/chat/regenerate")
async def regenerate_chat(req: RegenerateRequest, current_user: dict = Depends(get_current_user)):
    """
    Regenerate response for an existing query (updates existing record instead of creating new one)
    """
    user_id = current_user["user_id"]
    conversation_id = req.conversation_id or str(uuid.uuid4())
    
    # Verify the query exists and belongs to the user
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id,created_at FROM queries WHERE id=? AND user_id=?", (req.query_id, user_id))
    existing_query = c.fetchone()
    
    if not existing_query:
        conn.close()
        raise HTTPException(status_code=404, detail="Query not found")
    
    query_id, query_created_at = existing_query
    conn.close()

    safe_query = redact_pii(req.query)
    safe_query = sanitize_user_query(safe_query)
    start_time = time.time()

    # Initialize memory if missing
    conversation_memory[conversation_id] = []
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT query, answer FROM queries
        WHERE conversation_id=? AND project_id=? AND user_id=? AND created_at < ?
        ORDER BY created_at
    """, (conversation_id, req.project_id, user_id, query_created_at))
    rows = c.fetchall()
    conn.close()
    for q, a in rows:
        conversation_memory[conversation_id].append({"role": "user", "content": q})
        conversation_memory[conversation_id].append({"role": "assistant", "content": a})

    memory = conversation_memory[conversation_id]
    memory.append({"role": "user", "content": safe_query})

    # Initialize state for LangGraph
    initial_state = AgentState(
        user_query=safe_query,
        project_id=req.project_id,
        conversation_id=conversation_id,
        user_id=user_id,
        database_engine=req.database_engine,  
        memory=memory,
        query_intent="",
        next_steps=[],
        retrieval_mode="",
        required_documents=[],
        complexity="",
        keyword_search_words=[],
        generated_response="",
        contexts=[],
        attempt_count=0,
        is_fallback=False,
        current_step="start",
        rag_metrics={},
        validation_result="",
        needs_improvement=False,
        retry_target="none",
        improvement_suggestions=[],
        reflection_count=0
    )

    config = {
        "configurable": {
            "thread_id": conversation_id
        }
    }
    # Streaming response generator for regeneration
    async def stream_and_update():
        # Send initial status
        yield "[[[STATUS]]]🔄 Regenerating response...---SEPARATOR---"
        
        # Track previous status to avoid duplicates
        previous_status = ""

        # Stream status updates during workflow execution
        for current_state in agent_workflow.stream(initial_state, config=config):
            for node_name, state in current_state.items():
                if "current_status" in state and state["current_status"] != previous_status:
                    yield f"[[[STATUS]]]{state['current_status']}---SEPARATOR---"
                    await asyncio.sleep(0.01)  # allow flush to client
                    previous_status = state["current_status"]
            final_state = state
        
        workflow_state = final_state if final_state else agent_workflow.invoke(initial_state, config=config)
        
        # Get the final response
        full_answer = workflow_state.get("generated_response", "")
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Store assistant response in memory
        memory.append({"role": "assistant", "content": full_answer})
        
        # Truncate memory if needed
        if len(memory) > MAX_MEMORY_LENGTH:
            conversation_memory[conversation_id] = memory[-MAX_MEMORY_LENGTH:]
            
        rag_metrics = workflow_state.get("rag_metrics", {})
        
        # UPDATE existing query instead of creating new one
        conn = get_connection()
        c = conn.cursor()
        c.execute("""
            UPDATE queries 
            SET query = ?, answer = ?, response_time = ?, attempts = ?, is_fallback = ?,
                precision1 = ?, precision3 = ?, precision5 = ?, 
                claim_support = ?, static_exec = ?, query_coverage = ?, num_claims = ?,
                database_engine = ?
            WHERE id = ?
        """, (
            safe_query,  # Update the query (in case it was edited)
            full_answer, # Update the answer
            response_time,
            workflow_state.get("attempt_count", 1),
            workflow_state.get("is_fallback", False),
            rag_metrics.get("Precision@1"),
            rag_metrics.get("Precision@3"),
            rag_metrics.get("Precision@5"),
            rag_metrics.get("ClaimSupportRate"),
            rag_metrics.get("StaticExecScore"),
            rag_metrics.get("QueryCoverage"),
            rag_metrics.get("NumClaims"),
            req.database_engine, # Update the database engine if changed
            req.query_id  # The existing query ID
        ))
        conn.commit()
        conn.close()
        
        # Send the COMPLETE response at the end
        yield f"{full_answer}---SEPARATOR---"
        
        # Send metadata at the end (with the same query_id)
        meta = {
            "contexts": workflow_state.get("contexts", []),
            "query_id": req.query_id,
            "conversation_id": conversation_id
        }
        yield f"[[[META]]]{json.dumps(meta)}---SEPARATOR---"

    
    return StreamingResponse(stream_and_update(), media_type="text/plain")

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
    Fetch all queries (conversation history) for the user in a specific project.
    """
    conn = get_connection()
    c = conn.cursor()
    user_id = current_user["user_id"]
    
    try:
        # Convert project_id to integer and validate
        try:
            project_id_int = int(project_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid project ID")
        
        # Verify user has access to this project
        c.execute("""
            SELECT 1 FROM user_project 
            WHERE user_id = ? AND project_id = ?
        """, (user_id, project_id_int))
        
        if not c.fetchone():
            raise HTTPException(status_code=403, detail="Access denied to this project")
        
        # Fetch all queries from the database for this user and project
        c.execute(
            """
            SELECT id, conversation_id, project_id, query, answer, user_id, response_time, created_at, attempts, is_fallback, database_engine
            FROM queries
            WHERE user_id = ? AND project_id = ?
            ORDER BY created_at DESC
            """,
            (user_id, project_id_int),
        )
        rows = c.fetchall()

        conversations = [
            {
                "id": row[0],
                "conversation_id": row[1],
                "project_id": row[2],
                "query": row[3] or "",
                "answer": row[4] or "",
                "user_id": row[5],
                "response_time": row[6],
                "created_at": row[7],
                "attempts": row[8],
                "is_fallback": bool(row[9]) if row[9] is not None else False,
                "database_engine": row[10]
            }
            for row in rows
        ]

        return conversations

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching conversations: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error fetching conversations: {str(e)}"
        )
    finally:
        conn.close()

@app.get("/analytics-data")
def get_analytics_data(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Access forbidden: Admins only.")
 
    try:
        queries, feedback, projects, documents, users = load_data()
 
        # --- THIS IS THE FIX ---
        # Replace any NaN values in the queries DataFrame with None (which becomes 'null' in JSON)
        queries = queries.replace({np.nan: None})
       
        return {
            "queries": queries.to_dict(orient="records"),
            "feedback": feedback.to_dict(orient="records"),
            "projects": projects.to_dict(orient="records"),
            "documents": documents.to_dict(orient="records"),
            "users": users[["id", "username", "role"]].to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load analytics data: {str(e)}")
    

# Add Jira connection model
import subprocess
import asyncio
from contextlib import asynccontextmanager

# Global MCP process
mcp_process = None
mcp_client = None

class JiraConnectionRequest(BaseModel):
    jira_url: str
    username: str
    api_token: str
    is_default: bool = False

class JiraImportRequest(BaseModel):
    issue_key: str
    target_project_id: Optional[int] = None
    new_project_name: Optional[str] = None

class MCPJiraClient:
    """MCP Client for Jira with proper protocol handling"""
    
    def __init__(self, mcp_server_path: str = "jira_mcp_server.py"):
        self.mcp_server_path = mcp_server_path
        self.process = None
        self.connection_ids: dict[str, str] = {}
        self.initialized = False
        
    async def _read_stderr(self):
        while True:
            if not self.process:
                break
            line = await self.process.stderr.readline()
            if line is None:  # Process ended
                break
            if not line:  # Empty line
                continue
            print("MCP STDERR:", line.decode().strip())

    async def start_server(self):
        """Start MCP server"""
        try:
            if self.process and self.process.returncode is None:
                return 
            print("Starting MCP server...")
            import sys

            self.process = await asyncio.create_subprocess_exec(
                sys.executable,
                self.mcp_server_path,
                stdout=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            asyncio.create_task(self._read_stderr())
            print("MCP server started")
            
            # Initialize the connection
            await self._initialize_connection()
            
        except Exception as e:
            print(f"Failed to start MCP server: {e}")
            raise

    async def _initialize_connection(self):
        """Initialize MCP connection with proper handshake"""
        try:
            # Send initialize request
            init_id = str(uuid.uuid4())
            init_rpc = {
                "jsonrpc": "2.0",
                "id": init_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-02-01",
                    "capabilities": {
                        "tools": {}
                    },
                    "clientInfo": {
                        "name": "jira-fastapi-client",
                        "version": "1.0.0"
                    }
                }
            }
            
            print(f"SENDING INITIALIZE: {init_id}")
            init_line = json.dumps(init_rpc) + "\n"
            
            if self.process.stdin:
                self.process.stdin.write(init_line.encode())
                await self.process.stdin.drain()
                
                # Wait for initialize response
                init_response = await self._wait_for_response(init_id, timeout=10)
                if "error" in init_response:
                    print(f"Initialize failed: {init_response['error']}")
                    return False
                
                print("✅ MCP server initialized successfully")
                self.initialized = True
                
                # Optional: List tools to verify connection
                list_id = str(uuid.uuid4())
                list_rpc = {
                    "jsonrpc": "2.0",
                    "id": list_id,
                    "method": "tools/list",
                    "params": {}
                }
                
                print(f"SENDING TOOLS/LIST: {list_id}")
                list_line = json.dumps(list_rpc) + "\n"
                self.process.stdin.write(list_line.encode())
                await self.process.stdin.drain()
                
                # Wait for tools/list response
                list_response = await self._wait_for_response(list_id, timeout=10)
                if "error" not in list_response:
                    print("✅ Tools listed successfully")
                else:
                    print(f"Tools/list warning: {list_response.get('error', 'Unknown error')}")
                
                return True
            else:
                print("❌ Process stdin not available")
                return False
                
        except Exception as e:
            print(f"❌ Initialization error: {e}")
            return False

    async def _wait_for_response(self, req_id: str, timeout: int = 30) -> dict:
        """Wait for a specific response by ID"""
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            if self.process.stdout:
                try:
                    raw_line = await asyncio.wait_for(self.process.stdout.readline(), timeout=5)
                except asyncio.TimeoutError:
                    continue
                
                if raw_line is None:
                    print("❌ Process stdout returned None - process may have ended")
                    return {"error": "MCP server process ended unexpectedly"}
                        
                if not raw_line:
                    await asyncio.sleep(0.1)
                    continue

                line_str = raw_line.decode().strip()
                #print(f"RAW RESPONSE: {line_str}")
                
                if not line_str:
                    continue

                try:
                    msg = json.loads(line_str)
                except json.JSONDecodeError:
                    continue

                # Check if this is our response
                if msg.get("id") == req_id:
                    print(f"✅ MATCHED RESPONSE for {req_id}")
                    
                    if "error" in msg:
                        return {"error": msg["error"]}
                    
                    result = msg.get("result", {})
                    
                    # Extract content from result
                    content = result.get("content", [])
                    
                    if content:
                        # Handle different content formats
                        text_content = ""
                        for item in content:
                            if isinstance(item, dict):
                                if item.get("type") == "text":
                                    text_content = item.get("text", "")
                                elif "text" in item:
                                    text_content = item["text"]
                            elif isinstance(item, str):
                                text_content = item
                        
                        if text_content:
                            try:
                                parsed = json.loads(text_content)
                                return parsed
                            except json.JSONDecodeError:
                                return {"raw": text_content}
                    
                    return {"result": result}
        
        return {"error": "Timeout waiting for response"}

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call MCP tool with proper initialization"""
        print(f"🔧 Calling MCP tool: {tool_name}")
        print(f"📋 Arguments: {arguments}")
        
        try:
            # Ensure server is running and initialized
            if not self.process or self.process.returncode is not None:
                await self.start_server()
                
            # Check if process is still alive
            if self.process and self.process.returncode is not None:
                print("❌ MCP server process has terminated")
                return {"error": "MCP server process terminated"}
            
            # Ensure we're initialized
            if not self.initialized:
                print("🔄 Re-initializing connection...")
                if not await self._initialize_connection():
                    return {"error": "Failed to initialize MCP connection"}

            # Send tool call
            req_id = str(uuid.uuid4())
            rpc = {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            }
            
            print(f"📤 SENDING TOOL CALL: {tool_name} (ID: {req_id})")
            line = json.dumps(rpc) + "\n"
            
            if self.process.stdin:
                self.process.stdin.write(line.encode())
                await self.process.stdin.drain()
            else:
                return {"error": "Process stdin not available"}

            # Wait for response
            response = await self._wait_for_response(req_id, timeout=30)
            print(f"📥 TOOL RESPONSE: {response}")
            return response

        except Exception as e:
            print(f"❌ Error in call_tool: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    async def stop_server(self):
        """Stop MCP server"""
        if self.process:
            try:
                self.process.terminate()
                await self.process.wait()
                self.initialized = False
                print("MCP server stopped")
            except Exception as e:
                print(f"Error stopping MCP server: {e}")

# Initialize MCP client
mcp_jira_client = MCPJiraClient()


@app.on_event("startup")
async def startup():
    try:
        print("🔧 Initializing database...")
        init_db()   
        # Just start the server, initialization will happen on first tool call
        await mcp_jira_client.start_server()
        print("🔧 MCP server startup initiated")
    except Exception as e:
        print(f"🔧 startup failed: {e}")

# API Endpoints

@app.post("/jira/connect")
async def connect_jira(req: JiraConnectionRequest, current_user: dict = Depends(get_current_user)):
    """Connect to Jira via MCP"""
    user_id = current_user["user_id"]
    
    try:
        print(f"Connecting to Jira: {req.jira_url}")
        
        # Call MCP tool
        result = await mcp_jira_client.call_tool("connect_jira", {
            "jira_url": req.jira_url,
            "username": req.username,
            "api_token": req.api_token
        })
        print(f"MCP connect_jira result: {result}")
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        connection_id_mcp = result["connection_id"]
        mcp_jira_client.connection_ids[str(req.jira_url)] = connection_id_mcp
        
        # Save connection to database
        conn = get_connection()
        c = conn.cursor()
        
        if req.is_default:
            c.execute("UPDATE jira_connections SET is_default=0 WHERE user_id=?", (user_id,))
        
        c.execute("""
            INSERT INTO jira_connections (user_id, jira_url, username, api_token, is_default)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, req.jira_url, req.username, req.api_token, req.is_default))
        
        conn.commit()
        conn.close()
        
        return {"status": "connected", "message": "Successfully connected to Jira via MCP"}
        
    except Exception as e:
        print(f"Connection error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/jira/connections")
async def get_jira_connections(current_user: dict = Depends(get_current_user)):
    """Get user's Jira connections"""
    user_id = current_user["user_id"]
    conn = get_connection()
    c = conn.cursor()
    
    c.execute("SELECT id, jira_url, username, is_default, created_at FROM jira_connections WHERE user_id=?", (user_id,))
    connections = [{
        'id': row[0],
        'jira_url': row[1],
        'username': row[2],
        'is_default': bool(row[3]),
        'created_at': row[4]
    } for row in c.fetchall()]
    
    conn.close()
    return connections

@app.get("/jira/projects")
async def get_jira_projects(connection_id: int = Query(...), current_user: dict = Depends(get_current_user)):
    """Get projects from Jira via MCP"""
    user_id = current_user["user_id"]
    conn = get_connection()
    c = conn.cursor()
    
    c.execute("SELECT jira_url, username, api_token FROM jira_connections WHERE id=? AND user_id=?", 
              (connection_id, user_id))
    row = c.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Jira connection not found")
    
    jira_url, username, api_token = row
    conn.close()
    
    try:
        if jira_url in mcp_jira_client.connection_ids:
            connection_id_mcp = mcp_jira_client.connection_ids[jira_url]
        else:
            # Connect via MCP
            result = await mcp_jira_client.call_tool("connect_jira", {
                "jira_url": jira_url,
                "username": username,
                "api_token": api_token
            })
            
            if "error" in result:
                raise HTTPException(status_code=400, detail=result["error"])
            
            connection_id_mcp = result.get("connection_id")
            mcp_jira_client.connection_ids[jira_url] = connection_id_mcp
        
        # Get projects
        projects_result = await mcp_jira_client.call_tool("get_projects", {
            "connection_id": connection_id_mcp
        })
        print("GET_PROJECTS RAW:", projects_result)
        return projects_result.get("projects", [])
        
    except Exception as e:
        print(f"Error getting projects: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/jira/issues")
async def get_jira_issues(connection_id: int = Query(...), project_key: str = Query(...), current_user: dict = Depends(get_current_user)):
    """Get issues from a Jira project via MCP"""
    user_id = current_user["user_id"]
    conn = get_connection()
    c = conn.cursor()
    
    c.execute("SELECT jira_url, username, api_token FROM jira_connections WHERE id=? AND user_id=?", 
              (connection_id, user_id))
    row = c.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Jira connection not found")
    
    jira_url, username, api_token = row
    conn.close()
    
    try:
        if jira_url in mcp_jira_client.connection_ids:
            connection_id_mcp = mcp_jira_client.connection_ids[jira_url]
        else:
            # Connect via MCP
            result = await mcp_jira_client.call_tool("connect_jira", {
                "jira_url": jira_url,
                "username": username,
                "api_token": api_token
            })
            
            if "error" in result:
                raise HTTPException(status_code=400, detail=result["error"])
            
            connection_id_mcp = result.get("connection_id")
            mcp_jira_client.connection_ids[jira_url] = connection_id_mcp
        
        # Get issues
        issues_result = await mcp_jira_client.call_tool("get_issues", {
            "connection_id": connection_id_mcp,
            "project_key": project_key
        })
        
        return issues_result.get("issues", [])
        
    except Exception as e:
        print(f"Error getting issues: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/jira/import-issue")
async def import_jira_issue(req: JiraImportRequest, connection_id: int = Query(...), current_user: dict = Depends(get_current_user)):
    """Import a Jira issue into the application via MCP - attachments are optional"""
    user_id = current_user["user_id"]
    
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT jira_url, username, api_token FROM jira_connections WHERE id=? AND user_id=?", 
              (connection_id, user_id))
    row = c.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Jira connection not found")
    
    jira_url, username, api_token = row
    
    try:
        if jira_url in mcp_jira_client.connection_ids:
            connection_id_mcp = mcp_jira_client.connection_ids[jira_url]
        else:
            # Connect via MCP
            result = await mcp_jira_client.call_tool("connect_jira", {
                "jira_url": jira_url,
                "username": username,
                "api_token": api_token
            })
            
            if "error" in result:
                raise HTTPException(status_code=400, detail=result["error"])
            
            connection_id_mcp = result.get("connection_id")
            mcp_jira_client.connection_ids[jira_url] = connection_id_mcp
        
        # Get issue details
        issue_result = await mcp_jira_client.call_tool("get_issue", {
            "connection_id": connection_id_mcp,
            "issue_key": req.issue_key
        })
        
        if "error" in issue_result:
            raise HTTPException(status_code=404, detail=issue_result["error"])
        
        issue = issue_result.get("issue", {})
        
        # Determine target project
        target_project_id = req.target_project_id
        if not target_project_id and req.new_project_name:
            c.execute("INSERT INTO projects (name, created_by) VALUES (?, ?)", (req.new_project_name, user_id))
            target_project_id = c.lastrowid
            c.execute("INSERT INTO user_project (user_id, project_id, role) VALUES (?, ?, ?)", 
                     (user_id, target_project_id, 'user'))
            conn.commit()
        
        if not target_project_id:
            conn.close()
            raise HTTPException(status_code=400, detail="Either target_project_id or new_project_name must be provided")
        
        # Save imported issue (ALWAYS import the issue, even if attachments fail)
        attachments = issue.get('attachments', [])
        c.execute("""
            INSERT INTO jira_imported_issues 
            (project_id, jira_issue_key, jira_issue_summary, jira_issue_description, 
             jira_issue_type, jira_issue_status, jira_issue_assignee, jira_issue_attachments)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            target_project_id,
            issue.get('key'),
            issue.get('summary'),
            issue.get('description', ''),
            issue.get('issue_type'),
            issue.get('status'),
            issue.get('assignee', 'Unassigned'),
            json.dumps(attachments)
        ))
        imported_issue_id = c.lastrowid
        
        # Download and process attachments - BUT DON'T FAIL THE WHOLE IMPORT IF ATTACHMENTS FAIL
        successful_attachments = 0
        failed_attachments = []
        
        if attachments:
            proj_dir = os.path.join(RAW_DIR, str(target_project_id))
            os.makedirs(proj_dir, exist_ok=True)
            
            for attachment in attachments:
                print(f"🔄 Processing attachment: {attachment['filename']}")
                
                # Download via MCP
                download_result = await mcp_jira_client.call_tool("download_attachment", {
                    "connection_id": connection_id_mcp,
                    "attachment_url": attachment['url'],
                    "filename": attachment['filename']
                })
                
                # Check if download was successful
                if "error" not in download_result and download_result.get("status") == "success":
                    import base64
                    file_content_b64 = download_result.get("file_content")
                    original_name = attachment["filename"]
                    filename = original_name
                    filepath = os.path.join(proj_dir, filename)

                    counter = 1
                    name, ext = os.path.splitext(original_name)
                    while os.path.exists(filepath):
                        filename = f"{name} ({counter}){ext}"
                        filepath = os.path.join(proj_dir, filename)
                        counter += 1
                    attachment["filename"] = filename  # Update to final filename
                    
                    try:
                        with open(filepath, "wb") as f:
                            f.write(base64.b64decode(file_content_b64))
                        
                        # Save document reference even if we can't process it fully
                        c.execute("""
                            INSERT INTO documents (project_id, name, path, type, status, uploaded_by)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (target_project_id, attachment['filename'], filepath, "jira_attachment", "processing", user_id))
                        doc_id = c.lastrowid
                        conn.commit()
                        
                        # Process document in background
                        try:
                            documents = load_and_split_document(filepath)
                            
                            import threading
                            def background_index():
                                try:
                                    save_and_index(target_project_id, doc_id, attachment['filename'], filepath, "jira_attachment", documents)
                                    print(f"✅ Successfully indexed: {attachment['filename']}")
                                except Exception as e:
                                    print(f"⚠️ Indexing failed but file saved: {attachment['filename']} - {e}")
                                    # Update status to indicate partial failure
                                    c.execute("UPDATE documents SET status=? WHERE id=?", ("indexing_failed", doc_id))
                                    conn.commit()
                            thread = threading.Thread(target=background_index)
                            thread.daemon = True
                            thread.start()
                            
                        except Exception as processing_error:
                            print(f"⚠️ Processing failed but file saved: {attachment['filename']} - {processing_error}")
                            c.execute("UPDATE documents SET status=? WHERE id=?", ("processing_failed", doc_id))
                            conn.commit()
                        
                        successful_attachments += 1
                        
                    except Exception as e:
                        print(f"❌ Failed to save attachment: {attachment['filename']} - {e}")
                        failed_attachments.append(attachment['filename'])
                
                else:
                    print(f"🚫 Attachment Failed: {attachment['filename']}")
                    failed_attachments.append(attachment['filename'])
                    
                    # === DUPLICATE-SAFE RENAME FOR FAILED PLACEHOLDER ===
                    original_name = f"{attachment['filename']}"
                    filename = original_name
                    filepath = os.path.join(proj_dir, filename)

                    counter = 1
                    name, ext = os.path.splitext(original_name)
                    while os.path.exists(filepath):
                        filename = f"{name} ({counter}){ext}"
                        filepath = os.path.join(proj_dir, filename)
                        counter += 1
                        
                    attachment["filename"] = filename  # Update to final filename
                    
                    # Still create a placeholder document record for failed attachments
                    try:
                        placeholder_path = os.path.join(proj_dir, f"{attachment['filename']}")
                        with open(placeholder_path, "w") as f:
                            f.write(f"Attachment Failed: {attachment['url']}")
                        
                        c.execute("""
                            INSERT INTO documents (project_id, name, path, type, status, uploaded_by)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (target_project_id, f"{attachment['filename']}", placeholder_path, "jira_attachment", "failed", user_id))
                        conn.commit()
                    except Exception as placeholder_error:
                        print(f"⚠️ Could not create placeholder for failed attachment: {attachment['filename']}")

        conn.commit()
        conn.close()
        imported_attachments = [a["filename"] for a in attachments]

        return {
            "status": "imported",
            "project_id": target_project_id,
            "imported_issue_id": imported_issue_id,
            "issue_details": {
                "key": issue.get('key'),
                "summary": issue.get('summary'),
                "status": issue.get('status')
            },
            "attachments": imported_attachments,
            "attachments_summary": {
                "successful": successful_attachments,
                "failed": failed_attachments,
                "total": len(attachments)
            },
            "message": f"Successfully imported issue {issue.get('key')} - {successful_attachments}/{len(attachments)} attachments processed"
        }
        
    except Exception as e:
        conn.close()
        print(f"❌ Import error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
# ----------------------------
# Run with uvicorn
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_backend:app", host="127.0.0.1", port=8000, reload=True)
