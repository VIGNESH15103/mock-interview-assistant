import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

EMBEDDINGS_PATH = "data/processed/question_embeddings.npy"
METADATA_PATH = "data/processed/question_metadata.pkl"
INDEX_PATH = "data/processed/faiss_index.index"

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def retrieve_similar_questions(skill, top_k=10):
    query_embedding = model.encode([skill]).astype('float32')
    index, metadata = load_index()

    distances, indices = index.search(query_embedding, top_k * 3)  # Fetch more to fill all difficulties

    tree = {"Easy": [], "Medium": [], "Hard": []}
    for idx in indices[0]:
        if idx < len(metadata["questions"]):
            q = metadata["questions"][idx]
            d = metadata["difficulty"][idx]
            if d in tree and len(tree[d]) < top_k:
                tree[d].append(q)
        if all(len(tree[d]) >= top_k for d in tree):
            break

    return tree
