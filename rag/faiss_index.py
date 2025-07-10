import faiss
import numpy as np
import pickle
import os

EMBEDDINGS_PATH = "data/processed/question_embeddings.npy"
METADATA_PATH = "data/processed/question_metadata.pkl"
INDEX_PATH = "data/processed/faiss_index.index"

def build_faiss_index():
    print("🔍 Loading embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH).astype('float32')  # FAISS requires float32

    print("🧠 Creating FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    print("💾 Saving index to disk...")
    faiss.write_index(index, INDEX_PATH)

    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    print("✅ FAISS index built and saved.")
    return index, metadata

if __name__ == "__main__":
    build_faiss_index()
