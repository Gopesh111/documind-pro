# DocuMind Pro 🧠
### *Privacy-First RAG Engine with HNSW-Optimized Vector Search*

**DocuMind Pro** is a high-performance Retrieval-Augmented Generation (RAG) engine designed for secure, scalable document intelligence. Built for privacy-sensitive environments, it utilizes a custom-configured **HNSW (Hierarchical Navigable Small World)** index for sub-second retrieval across large datasets while enforcing a strict **Zero-Data-Retention** policy.

---

## 🚀 Key Engineering Highlights

* **Optimized Vector Search:** Implemented **FAISS HNSWFlat** indexing to achieve $O(\log N)$ search complexity, optimized for handling 10,000+ embeddings with minimal latency.
* **Privacy-Centric Middleware:** Engineered a custom **Zero-Data-Retention** pipeline that purges binary files and temporary session data from the server immediately after vectorization.
* **Production-Grade API:** Modular **FastAPI** backend featuring dependency injection, structured error handling, and environment isolation for enhanced security.
* **LLM Grounding:** Integrated with **Gemini-Pro** using specialized prompt engineering to ensure responses are strictly grounded in retrieved context, mitigating hallucination risks.

---

## 🛠️ Technical Stack

* **Language:** Python 3.10+
* **Vector Database:** FAISS (HNSW Indexing)
* **LLM Engine:** Google Gemini AI
* **Framework:** FastAPI (REST API)
* **Data Processing:** LangChain, NumPy, PyPDF
* **Testing:** PyTest (Security & Retrieval validation)

---

## 📂 Project Structure

    ├── api/            # REST API layer & Security Middleware
    ├── core/           # The Engine: HNSW Indexer, RAG Logic, & Processors
    ├── tests/          # Proof of Security & Retrieval Accuracy
    ├── main.py         # Application Entry Point
    └── .env.example    # Environment Isolation Template

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
    ```bash
    git clone git@github.com:yourusername/DocuMind-Pro.git
    cd DocuMind-Pro
    ```

2. **Set up Environment Variables:**
    Create a `.env` file based on `.env.example`:
    ```bash
    GEMINI_API_KEY=your_gemini_api_key_here
    TEMP_UPLOAD_DIR=./temp_uploads
    ```

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Engine:**
    ```bash
    python main.py
    ```
    *Access the interactive Swagger documentation at `http://localhost:8000/docs`*

---

## 🧪 Automated Verification (Tests)

The project includes a comprehensive test suite to validate the system's core claims:
* **Security Test:** Validates that the Zero-Retention middleware wipes all temporary data after the request lifecycle.
* **Retrieval Test:** Ensures the HNSW graph accurately identifies nearest neighbors with sub-second latency.

**Run tests using:**
    ```bash
    pytest -v
    ```

---

## 🛡️ Security Disclosure
This project uses **environment variable isolation** for all API credentials. It does not maintain a persistent database of uploaded documents; all vectors reside in volatile memory and are cleared upon session termination.

---
**Author:** Gopesh Pandey | B.Tech Computer Science (AI & ML) | 2022-2026 Batch