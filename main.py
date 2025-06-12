import streamlit as st
from utils.pdf_loader import load_pdf_chunks
from utils.vector_store import VectorStore
from agents.answer_agent import answer_agent
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Document Intelligence Assistant",
    page_icon="📘",
    layout="wide"
)

# Pro CSS Injection
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0f0f0f;
        color: #e6e6e6;
    }
    .main {
        padding: 2rem;
    }
    h1, h2, h3 {
        color: #00BFFF;
    }
    .stTextInput > div > div > input {
        border-radius: 8px;
        padding: 12px;
        background-color: #1c1c1e;
        color: #fff;
        border: 1px solid #444;
    }
    .stFileUploader {
        border: 2px dashed #555;
        border-radius: 12px;
        padding: 30px;
        background-color: #1c1c1e;
        transition: border 0.3s ease;
    }
    .stFileUploader:hover {
        border: 2px dashed #00BFFF;
    }
    .stButton > button {
        border-radius: 8px;
        padding: 10px 24px;
        background-color: #007acc;
        color: white;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #005fa3;
    }
    .success-box {
        background-color: #204d30;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        color: #d4edda;
        margin-top: 10px;
    }
    .answer-box {
        background-color: #002B5B;
        padding: 16px;
        border-radius: 10px;
        border-left: 4px solid #00BFFF;
        margin-top: 15px;
    }
    .processing-steps {
        background-color: #1a1a1a;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    hr {
        border: 0;
        height: 1px;
        background: #333;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://www.vecteezy.com/free-png/pdf-logo", width=100)
    st.markdown("## ⚙️ Settings")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
    top_k = st.slider("Top Results (k)", 1, 5, 3)
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("Mistral AI powered tool using RAG to answer your PDF queries.")

# Header
st.markdown("## 📘 Document Intelligence Assistant")
st.markdown("Upload a document below and ask questions based on its content.")

# File Upload
st.markdown("### 🗂️ Step 1: Upload your PDF")
uploaded_file = st.file_uploader("Choose a file", type="pdf", label_visibility="collapsed")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.markdown('<div class="success-box">✅ Document uploaded successfully!</div>', unsafe_allow_html=True)

    st.markdown("### 🔄 Processing Document")
    with st.container():
        st.markdown('<div class="processing-steps">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📄 Splitting**")
            with st.spinner("Splitting..."):
                chunks = load_pdf_chunks("temp.pdf")
            st.success("Done")

        with col2:
            st.markdown("**🧠 Indexing**")
            with st.spinner("Creating vector store..."):
                store = VectorStore(chunks)
            st.success("Ready")

        with col3:
            st.markdown("**🔍 Retrieval Setup**")
            st.success("System Ready ✅")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 🛠️ Current Configuration")
    st.json({
        "Chunk Size": chunk_size,
        "Top Chunks": top_k,
        "Embedding Model": "Mistral",
        "Retrieval Type": "Semantic RAG"
    })

    # Query section
    st.markdown("### 💬 Step 2: Ask Your Question")
    query = st.text_input(" ", placeholder="Ask something like: What is the document summary?", label_visibility="collapsed")

    if query:
        with st.spinner("Generating answer..."):
            try:
                top_chunks = store.search(query)
                context = "\n".join(top_chunks)
                answer = answer_agent(context, query)

                st.markdown("### ✅ Answer")
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                with st.expander("🔎 View Supporting Chunks"):
                    for i, chunk in enumerate(top_chunks, 1):
                        st.markdown(f"**Excerpt {i}**")
                        st.code(chunk, language="markdown")
                        st.markdown("---")
            except Exception as e:
                st.error(f"❌ Something went wrong: {str(e)}")

# Footer
st.markdown("""---  
<div style='text-align: center; font-size: 0.85rem; color: #777;'>  
    © 2025 Document Intelligence Assistant • Powered by 🧠 Mistral AI + LangChain  
</div>
""", unsafe_allow_html=True)
