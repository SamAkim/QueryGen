import json
import time
import pandas as pd
import streamlit as st
import requests
import uuid
from io import BytesIO
from streamlit_cookies_manager import EncryptedCookieManager
from app_backend import load_data
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

API_URL = "http://127.0.0.1:8000"

    
# ----------------------------
# CSS Styling
# ----------------------------

st.markdown("""
<style>
body {
    background-color: #E3D6EF;
}
.appview-container .main .block-container {
        padding-top: 0rem !important;  /* default is ~5rem, reduce as needed */
        padding-left: 2rem;
        padding-right: 2rem;
        padding-bottom: 0rem !important;
    }
/* Make all sidebar buttons borderless */
.stSidebar button {
    border: none;
    background-color: transparent;  /* optional: make background transparent too */
    box-shadow: none;  /* remove shadow if any */
    color: #3F0D66; 
}

/* Optional: remove hover effects */
.stSidebar button:hover {
    background-color: #3F0D66; 
    color:white; /* or keep transparent */
}

.project-button button {
    border: none !important;
    background-color: transparent !important;
    box-shadow: none !important;
    color: #3F0D66;
}

.project-button button:hover {
    background-color: #3F0D66 !important;
    color: white !important;
}

button:not(.project-list-btn):not(.stSidebar button) {
    background-color: #3F0D66 !important;
    color: white !important;
    border: none !important;
    box-shadow: none !important;
}

button:not(.project-list-btn):not(.stSidebar button):hover {
    opacity: 0.9;
    color:white;
    cursor: pointer;
}

.stChatInput { 
   border: 2px solid #3F0D66 !important;
   border-radius: 10px !important;
   padding: 5px !important;
}

.chat-container { max-height: 500px; overflow-y: auto; padding: 10px; margin-bottom: 10px; }
.user-bubble { background-color: #3F0D66; padding: 8px 12px; border-radius: 15px; margin: 5px; max-width: 100%; word-wrap: break-word; color:white; }
.user-message { justify-content: flex-end; }
.chat-message { display: flex; }

</style>
""", unsafe_allow_html=True)


# ----------------------------
# Session State Initialization
# ----------------------------
if "access_token" not in st.session_state:
    st.session_state.access_token = None

if "role" not in st.session_state:
    st.session_state.role = None
    
if "user_id" not in st.session_state:
    st.session_state.user_id = None
    
if "username" not in st.session_state:
    st.session_state.username = None
    
if "page_history" not in st.session_state:
    st.session_state.page_history = []

if "page" not in st.session_state:
    st.session_state.page = "login"
if "project_id" not in st.session_state:
    st.session_state.project_id = None
if "project_name" not in st.session_state:
    st.session_state.project_name = ""
if "chat_history" not in st.session_state or not isinstance(st.session_state.chat_history, dict):
    st.session_state.chat_history = {}
if "conversation_id" not in st.session_state or not isinstance(st.session_state.conversation_id, dict):
    st.session_state.conversation_id = {}
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []
if "doc_progress" not in st.session_state:
    st.session_state.doc_progress = {}
if "new_uploads" not in st.session_state:
    st.session_state.new_uploads = []
if "last_contexts" not in st.session_state:
    st.session_state.last_contexts = []
if "last_query_id" not in st.session_state:                    
    st.session_state.last_query_id = ""

# ----------------------------
# Initialize cookie manager
# ----------------------------
cookies = EncryptedCookieManager(
    prefix="myapp_",  # cookie prefix
    password="super-secret-password-123!",  # secret key to encrypt cookie
)
if st.session_state.page not in ["register", "login"] and not cookies.ready():
    st.stop()

def render_header_with_back():
    """Render global header and back button in same line."""
    cols = st.columns([2, 8])  # Adjust ratio for back button vs header

    with cols[0]:
        # Back button logic
        if st.session_state.page != "login" and st.session_state.page_history:
            if st.button("← Back"):
                while st.session_state.page_history:
                    prev_page = st.session_state.page_history.pop()
                    if prev_page != "login":
                        st.session_state.page = prev_page
                        st.rerun()

    with cols[1]:
        # Global header
        st.markdown(
            """
            <h3 style='
                text-align: center;  /* align left to stay next to back button */
                color: #3F0D66; 
                margin-top: -100px; 
                margin-bottom: 0px;
            '>🤖 AI-Powered SQL Automation</h3>
            """,
            unsafe_allow_html=True
        )
    
# ----------------------------
# Helper: Show User Info Banner
# ----------------------------
def show_user_banner():
    st.markdown(
        f"""
        <div style="position: absolute; top: 15px; right: 25px; font-size: 12px; color: gray;">
            🆔 User ID: <code>{st.session_state.user_id}</code>
        </div>
        """,
        unsafe_allow_html=True
    )
    
# ----------------------------
# Go to Page
# ----------------------------
def go_to_page(page_name, save_cookie=False):
    """
    Navigate to a page and optionally update last_page in cookies.
    - save_cookie: Only True when you really want to persist last_page.
    """
    if st.session_state.page != page_name:
        st.session_state.page_history.append(st.session_state.page)
    st.session_state.page = page_name
    st.rerun()
    
    if save_cookie:
        cookies["last_page"] = page_name  # save to cookie


# ----------------------------
# Helper: Set token cookie
# ----------------------------
def set_access_token(access_token, user_id, username):
    st.session_state.access_token = access_token
    st.session_state.user_id = user_id
    st.session_state.username = username
    cookies["access_token"] = access_token
    cookies["user_id"] = user_id
    cookies["username"] = username
    cookies["last_page"] = ""
    cookies.save()

# ----------------------------
# Helper: Clear token (logout)
# ----------------------------
def clear_access_token():
    st.session_state.access_token = None
    st.session_state.user_id = None
    st.session_state.username = None
    cookies["access_token"] = ""
    cookies["user_id"] = ""
    cookies["username"] = ""
    cookies.save()

# ----------------------------
# Attempt persistent login from cookie
# ----------------------------
if not st.session_state.access_token:
    if cookies.ready(): 
        access_token = cookies.get("access_token")
        if access_token:
            try:
                resp = requests.get(f"{API_URL}/verify_access_token",
                                    headers={"Authorization": f"Bearer {access_token}"})
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.user_id = data["user_id"]
                    st.session_state.username = data.get("username")
                    st.session_state.role = data.get("role", "user")
                    st.session_state.access_token = access_token
                    st.session_state.token_verified = True
                    last_page = cookies.get("last_page") or "query"
                    st.session_state.page = last_page
                    st.rerun()
                else:
                    clear_access_token()
            except Exception:
                clear_access_token()

# ----------------------------
# Login page
# ----------------------------
def login_page():
    render_header_with_back()
    col1, col2 = st.columns([7, 3])

    with col1:
        st.markdown("### 🔐 Login")  # Use markdown for smaller title if needed

    with col2:
        if st.button("New User → Register"):
            go_to_page("register")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username and password:
            try:
                resp = requests.post(f"{API_URL}/login", json={"username": username, "password": password})
                if resp.status_code == 200:
                    data = resp.json()
                    set_access_token(data["access_token"], data["user_id"], data["username"])
                    st.session_state.role = data.get("role", "user")
                    go_to_page("query")
                    st.session_state.project_id = None
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
            except Exception as e:
                st.error(f"❌ Login failed: {e}")
        else:
            st.error("❌ Enter username and password")

# ----------------------------
# Page: Register / Signup
# ----------------------------
def register_page():
    render_header_with_back()
    st.title("📝 Register")

    # --- Top-right link back to login ---
    col1, col2 = st.columns([8, 2])
    with col2:
        if st.button("← Back to Login"):
            go_to_page("login")

    # Input fields
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        if not username or not password or not confirm_password:
            st.error("❌ All fields are required.")
        elif password != confirm_password:
            st.error("❌ Passwords do not match.")
        else:
            try:
                # Call backend /register API
                resp = requests.post(
                    f"{API_URL}/register",
                    json={"username": username, "password": password},
                )
                if resp.status_code == 200:
                    st.success("✅ Registration successful! Please login.")
                    go_to_page("login")  # navigate to login page
                else:
                    st.error(f"❌ {resp.json().get('detail', 'Registration failed')}")
            except Exception as e:
                st.error(f"❌ Failed: {e}")

# ----------------------------
# Page: Projects
# ----------------------------
def projects_page():
    render_header_with_back()
    col1, col2 = st.columns([7, 3])  # Adjust widths as needed

    with col1:
        st.markdown("<h3 style='color: #3F0D66;'>📁 Projects</h3>", unsafe_allow_html=True)  # Use markdown instead of st.title()

    with col2:
        if st.button("➕ Create Project"):
            go_to_page("create_project")

    try:
        headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
        projects_resp = requests.get(f"{API_URL}/projects", headers=headers)
        projects = projects_resp.json() if projects_resp.status_code == 200 else []
    except Exception:
        projects = []


    if projects:
        st.markdown(
            """
        <style>
        .project-name {
            font-size: 16px;
            padding: 6px 10px;
            margin: 2px 0;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        for p in projects:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(
                    f"<div class='project-name' style='color: #3F0D66;'>📂 {p['project_name']}</div>",
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button("Open", key=f"open_{p['project_id']}"):
                    st.session_state.project_id = p["project_id"]
                    st.session_state.project_name = p["project_name"]
                    go_to_page("project_detail")
                    st.rerun()
    else:
        st.info("No projects found. Use ➕ Create Project above.")

# ----------------------------
# Page: Create Project
# ----------------------------
def create_project_page():
    render_header_with_back()
    st.title("➕ Create New Project")

    proj_name = st.text_input("Project Name")
    if st.button("Create"):
        if proj_name.strip():
            try:
                headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
                resp = requests.post(
                    f"{API_URL}/project",
                    json={"project_name": proj_name},
                    headers=headers,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.project_id = data["project_id"]
                    st.session_state.project_name = data["project_name"]
                    st.success(f"✅ Project '{data['project_name']}' created")
                    go_to_page("projects")
                else:
                    st.error(f"Project Name already exists. Please choose a different name.")
            except Exception as e:
                st.error(f"❌ Failed to create project: {e}")
        else:
            st.warning("Please enter a project name")

# ----------------------------
# Page: Project Detail
# ----------------------------
def project_detail_page():
    render_header_with_back()
    headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
    col1, col2 = st.columns([7, 3])
    with col1:
        st.markdown("<h3 style='color: #3F0D66; padding: 6px 10px; margin: 2px 0;'>📁 Documents</h3>", unsafe_allow_html=True)
        
    with col2:
        if st.button("➕ Add Documents"):
            go_to_page( "upload")

    try:
        docs_resp = requests.get(
            f"{API_URL}/project/{st.session_state.project_id}/documents",
            headers=headers,
        )
        documents = docs_resp.json().get("documents", []) if docs_resp.status_code == 200 else []
    except Exception:
        documents = []

    if documents:
        for doc in documents:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(
                    f"<div class='doc-name' style='color: #3F0D66;'>📓 {doc['doc_name']}</div>",
                    unsafe_allow_html=True,
                )
            if col2.button("🗑️ Delete", key=f"del_{doc['doc_id']}"):
                try:
                    resp = requests.delete(
                        f"{API_URL}/project/{st.session_state.project_id}/documents/{doc['doc_id']}",
                        headers=headers,
                    )
                    if resp.status_code == 200:
                        st.success(f"Deleted {doc['doc_name']} ✅")
                        st.session_state.uploaded_docs = [
                            d for d in st.session_state.uploaded_docs if d["doc_id"] != doc["doc_id"]
                        ]
                        st.rerun()
                    else:
                        st.error(f"❌ Failed: {resp.text}")
                except Exception as e:
                    st.error(f"❌ Exception: {e}")
    else:
        st.info("No documents uploaded yet.")

# ----------------------------
# Page: Upload Document
# ----------------------------
def upload_page():
    render_header_with_back()
    st.title("➕ Add Documents")

    uploaded_files = st.file_uploader(
        "Upload new documents (CSV, XLSX, PDF, DOCX)",
        type=["csv", "xlsx", "pdf", "docx", "txt"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("Upload"):
        try:
            headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
            files_payload = [
                ("files", (f.name, BytesIO(f.getvalue()), "application/octet-stream"))
                for f in uploaded_files
            ]
            response = requests.post(
                f"{API_URL}/upload",
                params={"project_id": st.session_state.project_id},
                files=files_payload,
                headers=headers,
            )
            results = response.json()
            for result in results:
                if result.get("status") == "processing":
                    st.success(f"Uploaded: {result['doc_name']}")
                    go_to_page("project_detail")
                else:
                    st.error(result.get("message", "Upload error"))
        except Exception as e:
            st.error(f"❌ Upload failed: {e}")

# ----------------------------
# Page: Query (Chat + Feedback)
# ----------------------------
def query_page():
    render_header_with_back()
    # --- 🔽 Project Selection Dropdown ---
    headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
    try:
        resp = requests.get(f"{API_URL}/projects", headers=headers)
        if resp.status_code == 200:
            projects = resp.json()
            project_names = {p["project_name"]: p["project_id"] for p in projects}

            # Default project selection
            current_proj_id = st.session_state.get("project_id")
            options = list(project_names.keys())
            
            # 🔹 Create 3 columns for dropdown, title, and button
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                st.markdown(
                    "<p style='margin-top: 8px; font-weight: 600; color: #3F0D66;'>Select Project:</p>",
                    unsafe_allow_html=True
                )

            with col2:
                selected_project = st.selectbox(
                    "Project",
                    options=options,
                    index=options.index(st.session_state.get("project_name"))
                    if st.session_state.get("project_name") in options else 0,
                    label_visibility="collapsed"   # hides "Project" label
                )

            with col3:
                if st.button("➕ New Query"):
                    st.session_state.chat_history[st.session_state.project_id] = []
                    st.session_state.conversation_id[st.session_state.project_id] = None
                    st.session_state.last_contexts = []
                    st.session_state.last_query_id = ""
                    st.rerun()

            # Update session_state if user changes project
            proj_id = project_names[selected_project]
            st.session_state.project_id = proj_id
            st.session_state.project_name = selected_project
        else:
            st.error("⚠️ Failed to load projects")
            return
    except Exception as e:
        st.error(f"❌ Could not fetch projects: {e}")
        return
    
    proj_id=st.session_state.project_id
    if not proj_id:
        st.warning("Please select a project first.")
        return

    if proj_id not in st.session_state.chat_history:
        st.session_state.chat_history[proj_id] = []
        
    chat_history = st.session_state.chat_history[proj_id]

# If chat history is empty, append default assistant greeting
    if not chat_history:
        chat_history.append({
            "role": "assistant",
            "content": "👋 Hi, I’m your SQL assistant. How can I help you today?"
        })

    if proj_id not in st.session_state.conversation_id:
        st.session_state.conversation_id[proj_id] = None

    chat_history = st.session_state.chat_history[proj_id]
    conversation_id = st.session_state.conversation_id[proj_id]

    # Show past messages using st.chat_message
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for idx, msg in enumerate(chat_history):
        if msg["role"] == "user":
            st.markdown(
                    f'<div class="chat-message user-message"><div class="user-bubble">{msg["content"]}</div></div>',
                    unsafe_allow_html=True,
                )
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
            default_greeting = "👋 Hi, I’m your SQL assistant. How can I help you today?"
            if msg["content"] != default_greeting:
                # Feedback only for assistant messages
                fb_col1, fb_col2, fb_col3 = st.columns([3, 1, 1])
                with fb_col1:
                    st.markdown("**Was this answer helpful?**")
                with fb_col2:
                    fb_up = st.button("👍", key=f"fb_up_{idx}")
                with fb_col3:
                    fb_down = st.button("👎", key=f"fb_down_{idx}")

                if f"feedback_type_{idx}" not in st.session_state:
                    st.session_state[f"feedback_type_{idx}"] = None
                if fb_up:
                    st.session_state[f"feedback_type_{idx}"] = "up"
                if fb_down:
                    st.session_state[f"feedback_type_{idx}"] = "down"

                if st.session_state[f"feedback_type_{idx}"] in ["up", "down"]:
                    comment = st.text_area("🗣️ Additional comment", key=f"fb_comment_{idx}", height=80)
                    if st.button("Submit Feedback", key=f"fb_submit_{idx}"):
                        feedback_payload = {
                            "query_id": msg.get("query_id", ""),
                            "project_id": st.session_state.project_id,
                            "query": msg.get("user_query", ""),
                            "answer": msg["content"],
                            "contexts": msg.get("contexts", []),
                            "feedback": st.session_state[f"feedback_type_{idx}"],
                            "comment": comment,
                            "user_id": st.session_state.user_id
                        }
                        try:
                            headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
                            resp = requests.post(f"{API_URL}/feedback", json=feedback_payload, headers=headers)
                            if resp.status_code == 200:
                                st.success("✅ Feedback submitted!")
                            else:
                                st.error(f"❌ Failed: {resp.text}")
                        except Exception as e:
                            st.error(f"❌ Failed: {e}")
    
    
    
    def send_query_streaming(payload, headers):
        try:
            with requests.post(f"{API_URL}/chat", json=payload, headers=headers, stream=True, timeout=300) as r:
                r.encoding = "utf-8"
                buffer = ""
                last_yield_len = 0  # to track what was already sent

                for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                    if not chunk:
                        continue

                    buffer += chunk

                    # ✅ detect metadata marker
                    if "[[[META]]]" in buffer:
                        text_part, meta_part = buffer.split("[[[META]]]", 1)
                        new_text = text_part[last_yield_len:]
                        if new_text.strip():
                            yield new_text  # send only new text
                        last_yield_len = len(text_part)

                        # parse metadata safely
                        try:
                            meta = json.loads(meta_part.strip())
                            st.session_state.last_contexts = meta.get("contexts", [])
                            st.session_state.last_query_id = meta.get("query_id", "")
                        except Exception as e:
                            st.warning(f"⚠️ Could not parse metadata: {e}")
                        break  # stop reading after metadata

                    else:
                        # stream only new incremental text
                        new_text = buffer[last_yield_len:]
                        if new_text:
                            yield new_text
                            last_yield_len = len(buffer)

        except Exception as e:
            st.error(f"❌ Streaming failed: {e}")
                
    # 🚀 Input is fixed at the bottom now
    if prompt := st.chat_input("Enter your Question..."):
        st.markdown(
                f'<div class="chat-message user-message"><div class="user-bubble">{prompt}</div></div>',
                unsafe_allow_html=True,
            )
        st.session_state.chat_history[proj_id].append({"role": "user", "content": prompt})
        payload = {
            "project_id": int(st.session_state.project_id),
            "query": prompt,
            "conversation_id": conversation_id,
        }  
        headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
        assistant_container = st.empty()  # placeholder for streaming
        spinner_container = st.empty()
        full_response = ""  
        # Status messages to show interactively
        
        with spinner_container:
            st.info("🤖 Generating response...")
        try:
            for chunk in send_query_streaming(payload, headers):
                full_response += chunk
                assistant_container.markdown(full_response)
                
            spinner_container.empty()

            # Save final assistant response in chat history
            st.session_state.chat_history[proj_id].append({
                "role": "assistant",
                "content": full_response,
                "user_query": prompt,
                "contexts": st.session_state.get("last_contexts", []),
                "query_id": st.session_state.get("last_query_id", "")
            })
            # Update conversation ID if returned from API (optional)
            st.session_state.conversation_id[proj_id] = conversation_id

            st.rerun()
        except Exception as e:
            spinner_container.empty()
            st.error(f"❌ Failed: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Page: Query History
# ----------------------------
def query_history():
    render_header_with_back()
    st_autorefresh(interval=10_000, key=f"query_history_conv")
    headers = {"Authorization": f"Bearer {st.session_state.access_token}"}

    # --- Select Project ---
    try:
        resp = requests.get(f"{API_URL}/projects", headers=headers)
        if resp.status_code == 200:
            projects = resp.json()
            project_names = {p["project_name"]: p["project_id"] for p in projects}
            options = list(project_names.keys())
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(
                    "<p style='margin-top: 8px; font-weight: 600; color: #3F0D66;'>Select Project:</p>",
                    unsafe_allow_html=True
                )
            with col2:
                selected_project = st.selectbox(
                    "Project",
                    options=options,
                    index=options.index(st.session_state.get("project_name"))
                    if st.session_state.get("project_name") in options else 0,
                    label_visibility="collapsed"
                )
            proj_id = project_names[selected_project]
            st.session_state.project_id = proj_id
            st.session_state.project_name = selected_project
        else:
            st.error("⚠️ Failed to load projects")
            return
    except Exception as e:
        st.error(f"❌ Could not fetch projects: {e}")
        return

    # --- Fetch Query History ---
    try:
        resp = requests.get(f"{API_URL}/queryhistory/{st.session_state.project_id}", headers=headers)
        if resp.status_code != 200:
            st.error("Failed to load Query history")
            return
        all_queries = resp.json()
        if not all_queries:
            st.info("No Query history available yet.")
            return

        selected_user = st.session_state.user_id
        user_convs = list({q.get("conversation_id") for q in all_queries if q.get("user_id") == selected_user})
        if not user_convs:
            st.info("No Queries found for this user.")
            return

        # Build dropdown options: only first query of each conversation
        conv_options = {}
        for conv_id in user_convs:
            conv_msgs = [q for q in all_queries if q.get("user_id") == selected_user and q.get("conversation_id") == conv_id]
            # get first query only
            first_query_msg = next((m.get("query") for m in sorted(conv_msgs, key=lambda x: x.get("created_at", ""))), "")
            if len(first_query_msg) > 80:
                first_query_msg = first_query_msg[:77] + "..."
            timestamp = max([m.get("created_at", "") for m in conv_msgs])
            display_text = f"{first_query_msg} ({timestamp})" if timestamp else first_query_msg
            conv_options[display_text] = timestamp,conv_id

        # Dropdown labels
        sorted_conv = sorted(conv_options.items(), key=lambda x: x[1][0], reverse=True)
        sorted_display = [x[0] for x in sorted_conv]

        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(
                "<p style='margin-top: 8px; font-weight: 600; color: #3F0D66;'>Select Conversation:</p>",
                unsafe_allow_html=True
            )
        with col2:
            selected_display = st.selectbox(
                "Select Conversation",
                options=sorted_display,
                index=0,
                label_visibility="collapsed",
                key=f"query_history_conv_select_{st.session_state.project_id}"
            )
            selected_conv = conv_options[selected_display][1]

        # --- Display full conversation in chronological order ---
        conv_history = [q for q in all_queries if q.get("user_id") == selected_user and q.get("conversation_id") == selected_conv]
        conv_history = sorted(conv_history, key=lambda x: x.get("created_at", ""))  # chronological

        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in conv_history:
            timestamp = msg.get("created_at", "")
            if msg.get("query"):
                st.markdown(
                    f'<div class="chat-message user-message"><div class="user-bubble">{msg["query"]}</div></div>',
                    unsafe_allow_html=True,
                )
            if msg.get("answer"):
                st.markdown(
                    msg["answer"]
                )
        st.markdown('</div>', unsafe_allow_html=True)

        # --- Continue Conversation button ---
        if st.button("Continue Conversation"):
            full_conversation = []
            for msg in conv_history:
                if msg.get("query"):
                    full_conversation.append({"role": "user", "content": msg["query"]})
                if msg.get("answer"):
                    full_conversation.append({"role": "assistant", "content": msg["answer"]})

            st.session_state.chat_history[proj_id] = full_conversation
            st.session_state.conversation_id[proj_id] = selected_conv

            go_to_page("query")
            st.rerun()

    except Exception as e:
        st.error(f"Error loading Query history: {e}")

        
        
def analytics_dashboard_page():
    st.set_page_config(page_title="📊 Analytics Dashboard", layout="wide")
    st.title("📊 SQL Assistant Analytics Dashboard")
    
    # Placeholder for live updates
    placeholder = st.empty()

    # Auto-refresh every 10 seconds
    st_autorefresh(interval=10_000, key="dashboard_refresh")

    with placeholder.container():
        queries, feedback, projects, documents, users = load_data()

        # Merge usernames into queries
        if "user_id" in queries.columns:
            queries = queries.merge(
                users[["id", "username"]],
                left_on="user_id",
                right_on="id",
                how="left"
            )

        # ---------------------------- Overview KPIs ----------------------------
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Projects", len(projects))
        col2.metric("Total Documents", len(documents))
        col3.metric("Total Queries", len(queries))
        col4.metric("Unique Conversations", queries["conversation_id"].nunique())
        col5.metric("Feedback Collected", len(feedback))
        st.markdown("---")

        # ---------------------------- Collapse/Expand Toggle ----------------------------
        expand_all = st.checkbox("🔽 Expand All Sections", value=True)

        # 🧑‍🤝‍🧑 User Analytics
        with st.expander("🧑‍🤝‍🧑 User Analytics", expanded=expand_all):
            if "username" in queries.columns:
                queries_per_user = queries.groupby("username")["query"].count().reset_index(name="query_count")
                fig = px.bar(queries_per_user, x="username", y="query_count", text_auto=True)
                st.plotly_chart(fig, use_container_width=True)

            sessions_per_user = queries.groupby("username")["conversation_id"].nunique().reset_index(name="session_count")
            fig_sessions = px.bar(sessions_per_user, x="username", y="session_count", text_auto=True)
            st.plotly_chart(fig_sessions, use_container_width=True)

            queries_per_session = queries.groupby("conversation_id")["query"].count()
            st.write(f"On average, users ask **{queries_per_session.mean():.2f} queries** per session.")

        # 📂 Query Analytics
        with st.expander("📂 Query Analytics", expanded=expand_all):
            queries_per_project = queries.groupby("project_id")["query"].count().reset_index(name="query_count")
            queries_per_project = queries_per_project.merge(
                projects[["id", "name"]],
                left_on="project_id",
                right_on="id",
                how="left"
            )
            fig_proj = px.bar(queries_per_project, x="name", y="query_count", color="query_count", text_auto=True)
            st.plotly_chart(fig_proj, use_container_width=True)

            conv_depth = queries.groupby("conversation_id")["query"].count()
            st.write(f"Average conversation depth: **{conv_depth.mean():.2f} queries**")
            fig_hist = px.histogram(conv_depth, nbins=10)
            st.plotly_chart(fig_hist, use_container_width=True)

            queries["created_at"] = pd.to_datetime(queries["created_at"])
            trend = queries.groupby(queries["created_at"].dt.date)["query"].count().reset_index()
            trend.columns = ["date", "total_queries"]
            fig_trend = px.line(trend, x="date", y="total_queries", markers=True)
            st.plotly_chart(fig_trend, use_container_width=True)

        # ⚡ Performance & Quality Metrics
        with st.expander("⚡ Performance & Quality Metrics", expanded=expand_all):
            col1, col2, col3 = st.columns(3)
            col4, col5, col6 = st.columns(3)
                
            if "precision1" in queries.columns:
                overall_accuracy = (queries["static_exec"].mean() + queries["claim_support"].mean()) / 2
                retrieval_quality = queries["precision1"].mean()
                query_understanding = queries["query_coverage"].mean()

                col1.metric("✅ Overall Accuracy", f"{overall_accuracy*100:.1f}%")
                col2.metric("📚 Retrieval Coverage", f"{retrieval_quality*100:.1f}%")
                col3.metric("🎯 Query Coverage", f"{query_understanding*100:.1f}%")
                
            if "response_time" in queries.columns:
                col4.metric("Avg Response Time (sec)", f"{queries['response_time'].mean():.2f}")

            if "attempts" in queries.columns:
                retry_rate = (queries["attempts"] > 1).mean() * 100
                col5.metric("Retry Rate", f"{retry_rate:.2f}%")

            if "is_fallback" in queries.columns:
                fallback_rate = queries["is_fallback"].mean() * 100
                col6.metric("Fallback Rate", f"{fallback_rate:.2f}%")

        # 👍👎 Feedback Analytics
        with st.expander("👍👎 Feedback Analytics", expanded=expand_all):
            if not feedback.empty:
                fb_counts = feedback["feedback"].value_counts().reset_index()
                fb_counts.columns = ["feedback_type", "count"]
                fig_fb = px.pie(fb_counts, names="feedback_type", values="count")
                st.plotly_chart(fig_fb, use_container_width=True)

        with st.expander("🔍 Conversation Explorer", expanded=expand_all):
            if "username" in queries.columns:
                selected_user = st.selectbox("Select a User", queries["username"].unique())
                user_convs = queries[queries["username"] == selected_user]["conversation_id"].unique()

                if len(user_convs) > 0:
                    selected_conv = st.selectbox("Select a Conversation", user_convs)

                    conv_history = queries[
                        (queries["username"] == selected_user) &
                        (queries["conversation_id"] == selected_conv)
                    ].sort_values("created_at")

                    st.subheader(f"Conversation {selected_conv} (User: {selected_user})")

                    for _, row in conv_history.iterrows():
                        with st.chat_message("user"):
                            st.markdown(row["query"])
                        with st.chat_message("assistant"):
                            st.markdown(row["answer"])
                else:
                    st.info("No conversations found for this user.")
            else:
                st.warning("No username column in queries table.")
        

# ----------------------------
# Router
# ----------------------------
# ----------------------------
# Sidebar Navigation (Streamlit native)
# ----------------------------
if st.session_state.page not in ["login", "register"]:
    with st.sidebar:
        # Always show user info + logout
        st.caption(f" 👤{st.session_state.username}")
        if st.button("🔓 Logout"):
            clear_access_token()
            go_to_page("login")
            st.rerun()

        # Show buttons based on current page
        if st.session_state.role == "admin":
            if st.button("📈 Analytics Dashboard"):
                go_to_page("analytics_dashboard")
                
        if st.session_state.page == "projects":
            if st.button("📂 Documents"):
                go_to_page("project_detail")
            if st.button("🗣️ Query"):
                go_to_page("query")
            if st.button("🔍 Query History"):
                go_to_page("query_history")

        if st.session_state.page == "create_project":
            if st.button("📁 Projects"):
                go_to_page("projects")
            if st.button("📂 Documents"):
                go_to_page("project_detail")
            if st.button("🗣️ Query"):
                go_to_page("query")
            if st.button("🔍 Query History"):
                go_to_page("query_history")

        elif st.session_state.page == "project_detail":
            if st.button("📁 Projects"):
                go_to_page("projects")
            if st.button("🗣️ Query"):
                go_to_page("query")
            if st.button("🔍 Query History"):
                go_to_page("query_history")

        elif st.session_state.page == "upload":
            if st.button("📁 Projects"):
                go_to_page("projects")
            if st.button("📂 Documents"):
                go_to_page("project_detail")
            if st.button("🗣️ Query"):
                go_to_page("query")
            if st.button("🔍 Query History"):
                go_to_page("query_history")

        elif st.session_state.page == "query":
            if st.button("📁 Projects"):
                go_to_page("projects")
            if st.button("📂 Documents"):
                go_to_page("project_detail")
            if st.button("🔍 Query History"):
                go_to_page("query_history")
                
        elif st.session_state.page == "query_history":
            if st.button("📁 Projects"):
                go_to_page("projects")
            if st.button("📂 Documents"):
                go_to_page("project_detail")
            if st.button("🗣️ Query"):
                go_to_page("query")
                
        elif st.session_state.page == "analytics_dashboard":
            if st.button("📁 Projects"):
                go_to_page("projects")
            if st.button("📂 Documents"):
                go_to_page("project_detail")
            if st.button("🗣️ Query"):
                go_to_page("query")
            if st.button("🔍 Query History"):
                go_to_page("query_history")


if st.session_state.page == "login":
    login_page()
if st.session_state.page == "register":
    register_page()
elif st.session_state.page == "projects":
    projects_page()
elif st.session_state.page == "create_project":
    create_project_page()
elif st.session_state.page == "project_detail":
    project_detail_page()
elif st.session_state.page == "query":
    query_page()
elif st.session_state.page == "upload":
    upload_page()
elif st.session_state.page == "analytics_dashboard":
    analytics_dashboard_page()
elif st.session_state.page == "query_history":  
    query_history()
