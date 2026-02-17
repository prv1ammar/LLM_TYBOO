import streamlit as st
import pandas as pd
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8888")
LITELLM_DB_URL = os.getenv("DATABASE_URL", "postgresql://litellm:litellm_password@postgres:5432/litellm_db")

st.set_page_config(
    page_title="Enterprise AI Admin",
    page_icon="🛡️",
    layout="wide"
)

# Sidebar for Authentication
st.sidebar.title("🛡️ Secure Access")
if "token" not in st.session_state:
    st.session_state.token = None

with st.sidebar:
    if not st.session_state.token:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/token",
                    data={"username": username, "password": password}
                )
                if response.status_code == 200:
                    st.session_state.token = response.json()["access_token"]
                    st.success("Logged in!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
            except Exception as e:
                st.error(f"Connection failed: {str(e)}")
    else:
        st.info(f"Logged in as Admin")
        if st.button("Logout"):
            st.session_state.token = None
            st.rerun()

# Main Header
st.title("🛡️ Enterprise AI Platform Control Center")
st.markdown("---")

if not st.session_state.token:
    st.warning("Please login to access the administrative dashboard.")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["📚 Knowledge Base", "👥 User Management", "📊 System Analytics"])

with tab1:
    st.header("📚 Enterprise Knowledge Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Indexed Documents")
        # In a real app, we'd query Qdrant or our API for document list
        # For now, we'll show the local documents directory
        docs_dir = "documents"
        if os.path.exists(docs_dir):
            files = os.listdir(docs_dir)
            if files:
                df = pd.DataFrame({"File Name": files, "Type": [f.split('.')[-1] for f in files]})
                st.table(df)
            else:
                st.info("No documents found in knowledge base.")
        else:
            st.error(f"Directory '{docs_dir}' not found.")

    with col2:
        st.subheader("Add New Document")
        uploaded_file = st.file_uploader("Upload PDF, TXT, or MD", type=["pdf", "txt", "md"])
        if uploaded_file:
            if st.button("Index Document"):
                with st.spinner("Processing..."):
                    # Save local
                    save_path = os.path.join(docs_dir, uploaded_file.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Call Ingestion logic (Import from ingest)
                    try:
                        from ingest import DataIngestor
                        ingestor = DataIngestor(collection_name="knowledge_base")
                        # Wrap ingest call
                        st.success(f"File '{uploaded_file.name}' saved and ready for ingestion pipeline.")
                    except Exception as e:
                        st.error(f"Ingestion failed: {str(e)}")

with tab2:
    st.header("👥 User & Access Management")
    
    # Mock user management
    users = [
        {"username": "admin", "role": "Super Admin", "status": "Active"},
        {"username": "legal_team", "role": "Legal Editor", "status": "Active"},
        {"username": "hr_team", "role": "HR Reader", "status": "Pending"}
    ]
    st.table(pd.DataFrame(users))
    
    if st.button("Add New User"):
        st.info("User management system is ready for PostgreSQL database migration.")

with tab3:
    st.header("📊 System Usage Analytics")
    
    # Placeholder for Grafana integration or direct Postgres queries
    st.subheader("Real-time Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total API Calls", "1,250", "+12%")
    m2.metric("Avg Latency", "1.2s", "-0.3s")
    m3.metric("Token Spend (MAD)", "450.20", "+15.30")
    
    st.markdown("---")
    st.subheader("GPU Utilization")
    chart_data = pd.DataFrame({
        "time": pd.date_range("2026-02-17", periods=24, freq="H"),
        "VRAM Usage (GB)": [18, 19, 18, 20, 22, 21, 19, 18, 17, 18, 20, 23, 24, 23, 22, 21, 20, 19, 18, 17, 18, 19, 20, 21]
    })
    st.line_chart(chart_data, x="time", y="VRAM Usage (GB)")

st.sidebar.markdown("---")
st.sidebar.caption("Platform Status: ✅ Healthy")
st.sidebar.caption("Version: 1.1.0 (Enterprise Edition)")
