import google.generativeai as genai
import os
from typing import List

class RAGEngine:
    def __init__(self, indexer):
        self.indexer = indexer
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-pro')

    def get_relevant_context(self, query: str) -> List[str]:
        # In a real 'Pro' app, you'd use a dedicated embedding model (e.g., HuggingFace)
        # For this example, we assume query_embedding is generated via Gemini/Vertex
        query_emb = self._gen_embedding(query)
        return self.indexer.search(query_emb)

    def generate_answer(self, query: str, context: List[str]) -> str:
        """
        Strict prompt engineering to prevent hallucinations.
        """
        context_str = "\n".join(context)
        prompt = f"""
        System: You are DocuMind Pro. Answer ONLY using the context below. 
        If the answer isn't there, say you don't know. 
        
        Context: {context_str}
        
        User Question: {query}
        """
        response = self.model.generate_content(prompt)
        return response.text

    def _gen_embedding(self, text: str):
        # Implementation for generating vector from text
        result = genai.embed_content(model="models/embedding-001", content=text)
        return np.array([result['embedding']])