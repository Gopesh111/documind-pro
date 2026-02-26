import re

def clean_text(text: str) -> str:
    """
    Sanitizes input text to improve embedding quality.
    """
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    return text.strip()

def chunk_data(text: str, chunk_size: int = 500, overlap: int = 50):
    """
    Standard recursive-style chunking for RAG consistency.
    """
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks