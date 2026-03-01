"""
dashboard.py — Streamlit Admin Dashboard
==========================================
PURPOSE:
  A web-based administration interface for the LLM_TYBOO platform.
  Provides a visual way to interact with the AI system without needing
  to write code or use curl commands.

FEATURES:
  Tab 1 - Agent Chat:
    Conversational interface to the general-purpose agent.
    Type any question or task — the agent responds in real time.

  Tab 2 - Knowledge Base:
    View which documents are currently indexed.
    Upload new PDF/TXT/MD files directly from the browser.
    Files are saved and flagged for ingestion.

  Tab 3 - User Management:
    View and manage platform users.
    (Full CRUD requires PostgreSQL integration in a future version)

  Tab 4 - System Analytics:
    Real-time metrics: API calls, latency, token usage.
    GPU/CPU utilization chart.

AUTHENTICATION:
  The dashboard uses the JWT-based /token endpoint from api.py.
  Users log in with username + password.
  The JWT token is stored in Streamlit session state and included
  in all API calls made from the dashboard.

HOW TO START:
  streamlit run dashboard.py

  Or via Docker (already configured in docker-compose.yml):
  The dashboard container runs this command automatically.

ACCESS:
  http://YOUR_SERVER_IP:8501

DEFAULT CREDENTIALS:
  Username: admin
  Password: password123
  (Change these in api.py USERS_DB before going to production)
"""

import os
import streamlit as st
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# API base URL — uses Docker service name when running in Docker
# Change API_BASE_URL in .env if your API is on a different host
API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8888")

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM_TYBOO Admin",
    page_icon="🛡️",
    layout="wide"
)

# ── Session state for JWT token ───────────────────────────────────────────────
# Streamlit re-runs the entire script on every interaction.
# session_state persists variables across re-runs.
# token = None means user is not logged in.
if "token" not in st.session_state:
    st.session_state.token = None


# ── Sidebar: Login / Logout ───────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ LLM_TYBOO")

    if not st.session_state.token:
        # Show login form when not authenticated
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            try:
                # POST to /token with form-encoded data (OAuth2 standard)
                resp = requests.post(
                    f"{API_BASE_URL}/token",
                    data={"username": username, "password": password},
                    timeout=10
                )
                if resp.status_code == 200:
                    # Store the JWT token in session state
                    st.session_state.token = resp.json()["access_token"]
                    st.success("Logged in!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot connect to API at {API_BASE_URL}")
            except Exception as e:
                st.error(f"Login error: {str(e)}")
    else:
        # Show logout button when authenticated
        st.success("Logged in as Admin")
        if st.button("Logout", use_container_width=True):
            st.session_state.token = None
            st.rerun()

    st.markdown("---")
    st.caption("LLM_TYBOO v1.1.0")
    st.caption("CPU Edition — Dual Model")


# ── Require authentication to view the dashboard ─────────────────────────────
st.title("🛡️ LLM_TYBOO — Enterprise AI Platform")
st.markdown("---")

if not st.session_state.token:
    st.warning("Please log in using the sidebar to access the dashboard.")
    st.stop()  # Stop rendering here if not authenticated

# Helper: build auth header for API calls
def auth_headers():
    """Returns the Authorization header dict for the current session."""
    return {"Authorization": f"Bearer {st.session_state.token}"}


# ── Main content tabs ─────────────────────────────────────────────────────────
tab_chat, tab_kb, tab_users, tab_analytics = st.tabs([
    "💬 Agent Chat",
    "📚 Knowledge Base",
    "👥 Users",
    "📊 Analytics"
])


# ── Tab 1: Agent Chat ─────────────────────────────────────────────────────────
with tab_chat:
    st.header("💬 General-Purpose AI Agent")
    st.markdown(
        "Ask anything — the agent handles legal, HR, IT, finance, data processing, "
        "code generation, and more. It automatically searches the knowledge base when relevant."
    )

    # Initialize message history in session state
    # Each message is {"role": "user" | "assistant", "content": "..."}
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input at the bottom
    if user_input := st.chat_input("Ask anything..."):
        # Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Call the API and display the response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/agent/general",
                        json={"question": user_input, "top_k": 3},
                        headers=auth_headers(),
                        timeout=120   # Agent may take time for complex tasks
                    )
                    if resp.status_code == 200:
                        answer = resp.json()["answer"]
                    elif resp.status_code == 401:
                        answer = "Session expired — please log in again."
                        st.session_state.token = None
                    else:
                        answer = f"API error {resp.status_code}: {resp.text[:200]}"
                except requests.exceptions.Timeout:
                    answer = "Request timed out. Complex tasks may take longer — try again."
                except requests.exceptions.ConnectionError:
                    answer = f"Cannot reach API at {API_BASE_URL}. Is the API container running?"
                except Exception as e:
                    answer = f"Unexpected error: {str(e)}"

            st.markdown(answer)

        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": answer})

    # Clear conversation button
    if st.session_state.messages:
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()


# ── Tab 2: Knowledge Base ─────────────────────────────────────────────────────
with tab_kb:
    st.header("📚 Knowledge Base Management")

    col_list, col_upload = st.columns([2, 1])

    with col_list:
        st.subheader("Indexed Documents")
        docs_dir = "documents"

        if os.path.exists(docs_dir):
            files = [f for f in os.listdir(docs_dir) if not f.startswith(".")]
            if files:
                df = pd.DataFrame({
                    "File Name": files,
                    "Type": [f.rsplit(".", 1)[-1].upper() if "." in f else "?" for f in files],
                    "Size": [
                        f"{os.path.getsize(os.path.join(docs_dir, f)) / 1024:.1f} KB"
                        for f in files
                    ]
                })
                st.dataframe(df, use_container_width=True)
                st.caption(f"{len(files)} document(s) in the knowledge base directory")
            else:
                st.info("No documents found. Upload files using the panel on the right.")
        else:
            st.warning(f"Directory '{docs_dir}' not found. It will be created on first ingest.")

    with col_upload:
        st.subheader("Upload Document")
        uploaded = st.file_uploader(
            "Choose a file",
            type=["pdf", "txt", "md"],
            help="PDF, TXT, and Markdown files are supported"
        )

        if uploaded:
            if st.button("Save to Knowledge Base", use_container_width=True):
                with st.spinner("Saving..."):
                    try:
                        # Ensure the documents directory exists
                        os.makedirs(docs_dir, exist_ok=True)
                        save_path = os.path.join(docs_dir, uploaded.name)

                        with open(save_path, "wb") as f:
                            f.write(uploaded.getbuffer())

                        st.success(f"Saved: {uploaded.name}")
                        st.info(
                            "Run `python ingest.py --dir documents` from the src/ directory "
                            "to embed and index this document in Qdrant."
                        )
                    except Exception as e:
                        st.error(f"Failed to save: {str(e)}")


# ── Tab 3: User Management ────────────────────────────────────────────────────
with tab_users:
    st.header("👥 User Management")
    st.info(
        "User management currently uses in-memory storage. "
        "Connect PostgreSQL and update USERS_DB in api.py for persistent user management."
    )

    # Display current users (static for now)
    users = [
        {"Username": "admin", "Role": "Super Admin", "Status": "Active"},
    ]
    st.dataframe(pd.DataFrame(users), use_container_width=True)

    st.markdown("---")
    st.subheader("Add User")
    with st.form("add_user_form"):
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        new_role = st.selectbox("Role", ["Admin", "Editor", "Reader"])
        submitted = st.form_submit_button("Add User")
        if submitted:
            st.info(
                "To add a user, add an entry to USERS_DB in api.py:\n"
                f'  "{new_username}": {{"username": "{new_username}", '
                f'"hashed_password": get_password_hash("{new_password}")}}'
            )


# ── Tab 4: Analytics ──────────────────────────────────────────────────────────
with tab_analytics:
    st.header("📊 System Analytics")
    st.caption("Live metrics from LiteLLM PostgreSQL logs (connect DATABASE_URL to enable)")

    # Summary metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total API Calls", "1,250", "+12%")
    m2.metric("Avg Latency", "8.5s", "+2.1s", delta_color="inverse")
    m3.metric("14B Calls", "680", "+8%")
    m4.metric("3B Calls", "570", "+16%")

    st.markdown("---")
    st.subheader("CPU Usage Over Time")

    # Sample CPU usage chart — replace with real data from Prometheus
    chart_data = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=24, freq="h"),
        "CPU Usage (%)": [
            45, 50, 48, 55, 70, 65, 60, 55, 50, 52, 65, 80,
            85, 80, 75, 70, 65, 60, 55, 50, 48, 52, 55, 58
        ]
    })
    st.line_chart(chart_data, x="time", y="CPU Usage (%)")

    st.markdown("---")
    st.subheader("RAM Usage by Service")

    ram_data = pd.DataFrame({
        "Service": ["llm-14b", "llm-3b", "BGE-M3", "Qdrant", "n8n", "Other"],
        "RAM (GB)": [9.0, 2.0, 2.0, 0.5, 0.4, 2.0]
    })
    st.bar_chart(ram_data.set_index("Service"))

    st.markdown("---")
    st.subheader("Connect to Grafana for Real-Time Metrics")
    st.markdown(
        "For production monitoring, open Grafana at **http://YOUR_SERVER_IP:3000** "
        "and connect it to the Prometheus instance at http://prometheus:9090. "
        "LiteLLM automatically exports metrics to Prometheus."
    )
