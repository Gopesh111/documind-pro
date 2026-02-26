import faiss
import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class DocumentIndexer:
    def __init__(self, dimension: int = 768, M: int = 32, ef_construction: int = 200):
        """
        Initializes an HNSW index for high-speed proximity search.
        M: Max number of connections per node (higher = more accurate, more memory).
        ef_construction: Search depth during index creation (higher = better index quality).
        """
        self.dimension = dimension
        # HNSW is a graph-based index superior to FlatL2 for 10k+ vectors
        self.index = faiss.IndexHNSWFlat(dimension, M)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = 64  # Depth of search during query time
        
        self.doc_map = {}  # Maps index IDs to text chunks
        logger.info(f"HNSW Index initialized with M={M}, ef={ef_construction}")

    def add_embeddings(self, embeddings: np.ndarray, text_chunks: List[str]):
        """
        Adds vectors to the HNSW graph.
        """
        if not self.index.is_trained:
            # HNSW doesn't require training like IVF, but good to check
            pass 
            
        start_id = self.index.ntotal
        self.index.add(embeddings.astype('float32'))
        
        for i, chunk in enumerate(text_chunks):
            self.doc_map[start_id + i] = chunk
            
        return {"count": len(text_chunks), "total": self.index.ntotal}

    def search(self, query_embedding: np.ndarray, k: int = 5):
        """
        Performs sub-second similarity search.
        """
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        results = [self.doc_map[idx] for idx in indices[0] if idx in self.doc_map]
        return results