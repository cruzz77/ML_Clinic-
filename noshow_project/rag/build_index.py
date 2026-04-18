import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from rag.guidelines import GUIDELINES

# Configuration
INDEX_PATH = "rag/faiss_index.bin"
METADATA_PATH = "rag/guidelines_metadata.pkl"

def build_faiss_index():
    print("Initializing embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Prepare documents
    documents = [f"{g['title']}: {g['content']}" for g in GUIDELINES]
    
    print(f"Encoding {len(documents)} guidelines...")
    embeddings = model.encode(documents)
    embeddings = np.array(embeddings).astype('float32')
    
    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save index and metadata
    os.makedirs("rag", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(GUIDELINES, f)
    
    print("FAISS index built and saved successfully.")

if __name__ == "__main__":
    build_faiss_index()
