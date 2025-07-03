import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings and metadata
embeddings = np.load("data/processed/question_embeddings.npy")
with open("data/processed/question_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

questions = metadata["questions"]

# Choose a test question
test_idx = 42
test_vector = embeddings[test_idx]
test_question = questions[test_idx]

# Compute similarity with all other questions
similarities = cosine_similarity([test_vector], embeddings)[0]
top_indices = similarities.argsort()[::-1][1:6]

print(f"\n🔍 Question: {test_question}")
print("\n🧠 Most similar questions:")
for i in top_indices:
    print(f"  • ({similarities[i]:.3f}) {questions[i]}")
