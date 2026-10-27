# GENQL — Agentic Text-to-SQL Studio

> **Enterprise-grade, model-agnostic SQL agent & RAG studio** that lets you chat with your databases in plain English, reason over documents, manage Jira issues, and monitor usage — all from a single, elegant interface.

---

## ✨ Features

### 🤖 Agentic Text-to-SQL
- Translate natural language questions into executable SQL using an LLM-powered agentic pipeline (LangGraph)
- Supports **PostgreSQL**, **MySQL**, **SQLite**, and more via SQLAlchemy
- **Multi-step reasoning**: the agent introspects your schema, plans queries, executes, and explains results
- Auto-retry with fallback on query errors
- Streaming responses for real-time feedback

### 📚 RAG (Retrieval-Augmented Generation)
- Upload and index **PDF**, **DOCX**, **CSV**, **images**, and more
- Documents are chunked, embedded (HuggingFace), and stored in a **FAISS** vector index per project
- Ask questions grounded in your uploaded knowledge base alongside SQL queries

### 🔌 Multi-Provider LLM Support
Switch between AI providers seamlessly — no code changes needed:

| Provider | Models |
|---|---|
| **Google Gemini** | `gemini-3.7-flash`, `gemini-3.5-flash`, and more |
| **OpenAI** | `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo` |
| **Anthropic** | `claude-3-5-sonnet`, `claude-3-5-haiku` |
| **Groq** | `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`, and more |
| **DeepSeek** | `deepseek-chat` |
| **Ollama** | Any locally-hosted model |

- **Automatic failover**: if the primary provider hits a quota limit, GENQL transparently switches to a backup

### 🗂️ Project Management
- Create and manage isolated **projects**, each with its own database connections, documents, and conversation history
- Role-based access: **admin**, **editor**, **user** per project
- Share projects with other registered users

### 🔗 Database Connections
- Connect to live databases (PostgreSQL, MySQL, SQLite) with credentials stored encrypted (Fernet)
- Multiple connections per project; switch between them at query time

### 🐛 Jira Integration (MCP)
- Built-in **MCP (Model Context Protocol) Jira server** (`jira_mcp_server.py`)
- Connect to any Jira instance; browse projects, search and filter issues
- Rich filtering by status, priority, assignee, label, dates, and more
- Summarise or query Jira issues through the same chat interface

### 📊 Analytics Dashboard (Streamlit)
- Admin-only dashboard (`analytics_dashboard.py`) built with **Streamlit + Plotly**
- Tracks: total queries, unique conversations, user activity, feedback, project usage
- Visualises query latency, precision metrics, and feedback scores over time

### 🔐 Auth & Security
- JWT-based authentication (HS256) with access tokens and refresh support
- bcrypt password hashing with configurable rounds
- Rate-limited login with lockout after failed attempts
- Role-based access control (RBAC): `admin` / `user`
- Database credentials encrypted at rest with Fernet

---

## 🏗️ Architecture

```
GENQL/
├── index.html            # Single-page frontend (HTML + Vanilla JS + CSS)
├── script.js             # Frontend logic: auth, chat, Jira, DB UI
├── style.css             # Dark-mode design system
├── app_backend.py        # FastAPI backend — API routes, DB logic, RAG, SQL agent
├── backend_agents.py     # LangGraph agent definitions & async workflow engine
├── llm_provider.py       # Unified multi-provider LLM abstraction layer
├── auth.py               # JWT auth router (login, register, token verify)
├── jira_mcp_server.py    # MCP-compliant Jira integration server (stdio)
├── analytics_dashboard.py# Streamlit admin analytics dashboard
├── requirements.txt      # Python dependencies
├── .env                  # Environment configuration (see below)
├── app_data.db           # SQLite application database (auto-created)
├── data/                 # Uploaded files (raw) & intermediate processing
└── stores/               # FAISS vector stores (per project)
```

### Key Technology Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, Vanilla JS, CSS3 (dark glassmorphism) |
| Backend | FastAPI + Uvicorn |
| AI Orchestration | LangGraph (StateGraph), LangChain |
| Vector Store | FAISS + HuggingFace Embeddings |
| Database | SQLite (app state), SQLAlchemy (user DBs) |
| Auth | JWT (python-jose), bcrypt |
| Analytics | Streamlit + Plotly |
| Jira | MCP Server (stdio) |
| Document Parsing | PyMuPDF, python-docx, unstructured |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip or a virtual environment manager

### 1. Clone the repository

```bash
git clone <repo-url>
cd GENQL
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example below into a `.env` file in the project root and fill in your values:

```env
# LLM Provider
# Options: gemini | openai | anthropic | groq | deepseek | ollama
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.7-flash

# API Keys — set whichever provider(s) you want to use
GEMINI_API_KEY="your-gemini-api-key"
# OPENAI_API_KEY="sk-..."
# ANTHROPIC_API_KEY="sk-ant-..."
# GROQ_API_KEY="gsk_..."
# DEEPSEEK_API_KEY="sk-..."
# OLLAMA_BASE_URL="http://localhost:11434/v1"

# Security
SECRET_KEY=change-me-in-production-use-a-long-random-string

# Admin Credentials
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin123

# Auth Settings
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
MAX_LOGIN_ATTEMPTS=5
LOCKOUT_TIME_MINUTES=15
BCRYPT_ROUNDS=12

# Optional
# JIRA_SSL_VERIFY=true
```

> **Never commit your real `.env` file to version control.**

### 5. Start the backend

```bash
uvicorn app_backend:app --reload --host 127.0.0.1 --port 8000
```

The backend serves the frontend statically. Open your browser at:

```
http://127.0.0.1:8000
```

### 6. (Optional) Start the Analytics Dashboard

```bash
streamlit run analytics_dashboard.py
```

### 7. (Optional) Start the Jira MCP Server

The Jira server communicates over **stdio** and is launched by the backend automatically when Jira features are used. To run it manually for testing:

```bash
python jira_mcp_server.py
```

---

## 🔑 Default Credentials

| Role | Username | Password |
|---|---|---|
| Admin | `admin` | `admin123` |

> Change these immediately via your `.env` file before any deployment.

---

## 📡 API Overview

The FastAPI backend exposes a REST API at `http://127.0.0.1:8000`. Interactive docs are available at:

- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`

### Auth endpoints (`/auth/`)

| Method | Path | Description |
|---|---|---|
| `POST` | `/auth/login` | Authenticate and receive a JWT |
| `POST` | `/auth/register` | Register a new user account |
| `GET` | `/auth/verify_access_token` | Validate a JWT and return user info |

### Core endpoints (authenticated)

| Area | Description |
|---|---|
| Projects | Create, list, share, and delete projects |
| Documents | Upload, process, and manage RAG documents |
| Query | Submit natural-language questions → SQL + answer |
| Database | Manage live database connections (CRUD) |
| Conversations | List, fetch, and delete conversation history |
| Feedback | Submit thumbs up/down feedback on answers |
| Jira | Connect, browse projects, and search issues |

---

## 🛡️ Security Notes

- All database passwords are **Fernet-encrypted** before being stored in SQLite
- User passwords are hashed with **bcrypt** (configurable rounds)
- JWT tokens use **HS256** with a secret key you must set in `.env`
- The admin account is created on first startup from `.env` credentials — keep them secret
- CORS is configured — review `CORSMiddleware` in `app_backend.py` before deploying publicly

---

## 📦 Key Dependencies

See [`requirements.txt`](./requirements.txt) for the full list.

| Package | Purpose |
|---|---|
| `fastapi` + `uvicorn` | Web server & API |
| `langgraph` + `langchain` | Agent orchestration |
| `faiss-cpu` | Vector similarity search |
| `langchain-huggingface` | Embeddings |
| `google-genai` | Gemini AI provider |
| `sqlalchemy` + `psycopg2-binary` + `pymysql` | Database connectors |
| `PyMuPDF` + `python-docx` | Document parsing |
| `streamlit` + `plotly` | Analytics dashboard |
| `mcp` | Jira MCP server protocol |
| `python-jose` + `bcrypt` | Auth & security |
| `cryptography` | Credential encryption |
| `sqlglot` | SQL dialect translation |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "feat: add my feature"`
4. Push and open a Pull Request

---

## 📄 License

This project is provided as-is. Add your preferred license here.
