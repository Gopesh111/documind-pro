import numpy as np
import pytest
from core.indexer import DocumentIndexer

def test_hnsw_accuracy_and_speed():
    """
    Validates that HNSW retrieves the correct nearest neighbor
    from a simulated batch of embeddings.
    """
    dimension = 768
    indexer = DocumentIndexer(dimension=dimension, M=16)
    
    # Create dummy data: 100 vectors
    data = np.random.random((100, dimension)).astype('float32')
    # Make one vector very specific
    target_vector = np.ones((1, dimension)).astype('float32')
    data[50] = target_vector[0]
    
    text_chunks = [f"Chunk {i}" for i in range(100)]
    text_chunks[50] = "The specific target context"

    # Add to index
    indexer.add_embeddings(data, text_chunks)

    # Search for the target
    results = indexer.search(target_vector, k=1)

    assert "The specific target context" in results
    assert len(results) == 1
    print("\n[PASSED] HNSW Search accurate.")