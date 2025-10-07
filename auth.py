# auth.py
import os
import sqlite3
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt
import uuid

# ----------------------------
# DB connection
# ----------------------------
DB_PATH = os.path.join(os.path.dirname(__file__), "app_data.db")
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

# ----------------------------
# Create tables if not exists
# ----------------------------
conn = get_connection()
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE,
    password_hash TEXT,
    role TEXT DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
conn.commit()
conn.close()

# ----------------------------
# Password hashing
# ----------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)

# ----------------------------
# JWT utils
# ----------------------------
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 30

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(access_token: str):
    try:
        payload = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

# ----------------------------
# FastAPI router
# ----------------------------
router = APIRouter()
security = HTTPBearer()

# ----------------------------
# Pydantic models
# ----------------------------
class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    role: str = "user"  # optional: 'user' or 'admin'

# ----------------------------
# Login endpoint
# ----------------------------
@router.post("/login")
def login(req: LoginRequest):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (req.username,))
    row = c.fetchone()
    conn.close()
    if not row or not verify_password(req.password, row[3]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token({"user_id": row[0], "role": row[4], "username": row[2]})
    return {"access_token": access_token, "role": row[4], "user_id": row[0], "username": row[2]}

# ----------------------------
# Register endpoint
# ----------------------------
@router.post("/register")
def register(req: RegisterRequest):
    user_id = str(uuid.uuid4())
    hashed_password = hash_password(req.password)
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (id, username, password_hash, role) VALUES (?, ?, ?, ?)",
            (user_id, req.username, hashed_password, req.role)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=400, detail="Username already exists")
    conn.close()
    return {"user_id": user_id, "username": req.username, "role": req.role}


# ----------------------------
# Dependency for protected routes
# ----------------------------
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    access_token = credentials.credentials
    payload = decode_access_token(access_token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return {"user_id": payload["user_id"], "role": payload["role"], "username": payload.get("username")}

# ----------------------------
# Create initial admin if not exists
# ----------------------------
def create_initial_admin():
    admin_username = os.getenv("ADMIN_USERNAME", "admin")
    admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username=?", (admin_username,))
    if not c.fetchone():
        hashed_password = hash_password(admin_password)
        c.execute(
            "INSERT INTO users (id, username, password_hash, role) VALUES (?, ?, ?, ?)",
            (str(uuid.uuid4()), admin_username, hashed_password, "admin")
        )
        conn.commit()
    conn.close()


@router.get("/verify_access_token")
def verify_access_token(current_user: dict = Depends(get_current_user)):
    """
    Endpoint to verify JWT token from frontend cookie.
    Returns user info if valid, else 401.
    """
    return {
        "user_id": current_user["user_id"],
        "role": current_user["role"],
        "username": current_user.get("username")
    }
