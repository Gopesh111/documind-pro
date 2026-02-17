import streamlit as st
import os, tempfile, time, re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="DocuMind Pro",
    page_icon="📄",
    layout="wide"
)

# ==========================
# CLEAN SAAS UI
# ==========================
st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
    color: #e5e7eb;
}
section[data-testid="stSidebar"] {
    background-color: #020617;
    border-right: 1px solid #1e293b;
}
h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #38bdf8;
}
.subtitle {
    color: #94a3b8;
    font-size: 1rem;
    margin-top: -8px;
}
.block {
    background-color: #020617;
    padding: 24px;
    border-radius: 12px;
    border: 1px solid #1e293b;
}
.stButton>button {
    background-color: #38bdf8;
    color: #020617;
    font-weight: 600;
    border-radius: 8px;
}
[data-testid="stChatMessage"] {
    background-color: #020617;
    border: 1px solid #1e293b;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# HEADER
# ==========================
st.info("🔐 Each session is isolated. Uploaded PDFs are processed in-memory and never shared.")

st.markdown("<h1>DocuMind Pro</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Chat with your PDFs. Compare documents. Get grounded answers.</div>",
    unsafe_allow_html=True
)
st.markdown("---")

# ==========================
# CONFIG
# ==========================
API_KEY = os.getenv("GOOGLE_API_KEY")


# ==========================
# SIDEBAR
# ==========================

with st.sidebar:
    st.markdown("### 📂 Workspace")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    st.markdown("---")
    MODE = st.radio(
        "Interaction Mode",
        ["Single Knowledge Base", "Multi-PDF Comparison"]
    )

    st.markdown("---")
    RESPONSE_MODE = st.selectbox(
        "Response Style",
        ["Concise", "Detailed", "Technical", "Executive Summary"]
    )

    st.markdown("---")
    if st.button("Reset Session"):
        st.session_state.clear()
        st.rerun()
with st.sidebar:
    st.caption("🔒 Privacy-first: Files stay in memory and are deleted when the session ends.")


# ==========================
# EMBEDDINGS
# ==========================
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=API_KEY
)

# ==========================
# HELPERS — SENTENCE CITATION
# ==========================
def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)

def sentence_level_citations(answer, source_docs):
    sentences = split_sentences(answer)
    source_texts = [d.page_content for d in source_docs]
    source_meta = [d.metadata for d in source_docs]

    sent_emb = embeddings.embed_documents(sentences)
    src_emb = embeddings.embed_documents(source_texts)

    final_output = []

    for i, s_emb in enumerate(sent_emb):
        sims = cosine_similarity([s_emb], src_emb)[0]
        best_idx = np.argmax(sims)
        meta = source_meta[best_idx]

        doc = meta.get("source", "Document")
        page = meta.get("page", "N/A")

        final_output.append(
            f"{sentences[i]}\n\n*(Source: {doc} · Page {page})*"
        )

    return "\n\n".join(final_output)

# ==========================
# VECTOR BUILDERS
# ==========================
@st.cache_resource
def build_single_db(files):
    docs = []
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(f.getvalue())
            path = tmp.name

        loaded = PyPDFLoader(path).load()
        for d in loaded:
            d.metadata["source"] = f.name

        docs.extend(loaded)
        os.remove(path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, embeddings)

@st.cache_resource
def build_multi_db(files):
    dbs = {}
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    for f in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(f.getvalue())
            path = tmp.name

        docs = PyPDFLoader(path).load()
        for d in docs:
            d.metadata["source"] = f.name

        chunks = splitter.split_documents(docs)
        dbs[f.name] = FAISS.from_documents(chunks, embeddings)
        os.remove(path)

    return dbs

# ==========================
# MAIN
# ==========================
if not uploaded_files:
    st.markdown(
        "<div class='block'>👋 Upload PDFs to start asking questions.</div>",
        unsafe_allow_html=True
    )

else:
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        google_api_key=API_KEY,
        temperature=0.2
    )

    # ==========================
    # SINGLE KB MODE
    # ==========================
    if MODE == "Single Knowledge Base":
        db = build_single_db(uploaded_files)
        retriever = db.as_retriever(search_kwargs={"k": 5})

        if query := st.chat_input("Ask something about your documents"):
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                docs = retriever.invoke(query)
                context = "\n\n".join(d.page_content for d in docs)

                prompt = f"""
Answer using ONLY the context below.
Response style: {RESPONSE_MODE}

Context:
{context}

Question:
{query}
"""
                raw_answer = llm.invoke(prompt).content
                cited_answer = sentence_level_citations(raw_answer, docs)

                st.markdown(cited_answer)

    # ==========================
    # MULTI PDF MODE
    # ==========================
    else:
        dbs = build_multi_db(uploaded_files)
        query = st.text_input("Ask one question across all documents")

        if query:
            cols = st.columns(len(dbs))
            for col, (name, db) in zip(cols, dbs.items()):
                with col:
                    st.markdown(f"**{name}**")
                    retriever = db.as_retriever(search_kwargs={"k": 4})
                    docs = retriever.invoke(query)

                    context = "\n\n".join(d.page_content for d in docs)
                    prompt = f"""
Answer using ONLY this document.
Response style: {RESPONSE_MODE}

Context:
{context}

Question:
{query}
"""
                    raw_answer = llm.invoke(prompt).content
                    cited_answer = sentence_level_citations(raw_answer, docs)

                    st.markdown(cited_answer)
