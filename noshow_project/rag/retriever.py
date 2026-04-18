import faiss
import pickle
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# Configuration
INDEX_PATH = "noshow_project/rag/faiss_index.bin"
METADATA_PATH = "noshow_project/rag/guidelines_metadata.pkl"

# Singleton model and index holders
_MODEL = None
_INDEX = None
_GUIDELINES = None

def get_resources():
    global _MODEL, _INDEX, _GUIDELINES
    
    # Check if we need to adjust paths for local vs app run
    idx_path = INDEX_PATH if os.path.exists(INDEX_PATH) else "rag/faiss_index.bin"
    meta_path = METADATA_PATH if os.path.exists(METADATA_PATH) else "rag/guidelines_metadata.pkl"

    if _MODEL is None:
        _MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    
    if _INDEX is None and os.path.exists(idx_path):
        _INDEX = faiss.read_index(idx_path)
    
    if _GUIDELINES is None and os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            _GUIDELINES = pickle.load(f)
            
    return _MODEL, _INDEX, _GUIDELINES

def retrieve_guidelines(query: str, top_k: int = 3) -> str:
    """
    Search guidelines based on query and return formatted context.
    """
    try:
        model, index, guidelines = get_resources()
        
        if index is None or guidelines is None:
            return "Knowledge base not initialized. Using default protocols."
            
        # Encode query
        query_vec = model.encode([query]).astype('float32')
        
        # Search index
        distances, indices = index.search(query_vec, top_k)
        
        # Format results
        results = []
        for idx in indices[0]:
            if idx < len(guidelines):
                doc = guidelines[idx]
                results.append(f"### {doc['title']}\n{doc['content']}")
        
        return "\n\n".join(results)
    except Exception as e:
        return f"Error in RAG retrieval: {str(e)}"
