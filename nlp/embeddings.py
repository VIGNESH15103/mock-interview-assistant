from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import pickle
import os

DATA_PATH = "data/processed/cleaned_questions.csv"
EMBEDDINGS_PATH = "data/processed/question_embeddings.npy"
METADATA_PATH = "data/processed/question_metadata.pkl"

def generate_embeddings():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Cleaned data not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    questions = df["Question"].tolist()
    ids = df["ID"].tolist()

    print("🔍 Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("🔧 Generating embeddings...")
    embeddings = model.encode(questions, show_progress_bar=True)

    print("💾 Saving embeddings and metadata...")
    np.save(EMBEDDINGS_PATH, embeddings)

    metadata = {
        "ids": ids,
        "questions": questions,
        "categories": df["Category"].tolist(),
        "difficulty": df["Difficulty"].tolist(),
    }
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("✅ Embeddings saved!")

if __name__ == "__main__":
    generate_embeddings()
