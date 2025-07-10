import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDINGS_PATH = "data/processed/question_embeddings.npy"
METADATA_PATH = "data/processed/question_metadata.pkl"

def load_embeddings():
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return embeddings, metadata

def build_dynamic_tree(category, embeddings, metadata, top_k=3):
    tree = {"Easy": [], "Medium": [], "Hard": []}

    # Filter indices for the chosen category
    indices = [i for i, c in enumerate(metadata["categories"]) if c == category]

    if not indices:
        print(f"⚠️ No questions found for category: {category}")
        return tree

    # Extract relevant embeddings and metadata for the category
    cat_embeddings = embeddings[indices]
    cat_questions = [metadata["questions"][i] for i in indices]
    cat_difficulties = [metadata["difficulty"][i] for i in indices]

    # Compute the centroid for semantic center
    centroid = np.mean(cat_embeddings, axis=0, keepdims=True)
    scores = cosine_similarity(cat_embeddings, centroid).flatten()

    # Sort by similarity to the centroid
    sorted_indices = np.argsort(scores)[::-1]

    # Fill tree based on difficulty
    for i in sorted_indices:
        diff = cat_difficulties[i]
        if diff in tree and len(tree[diff]) < top_k:
            tree[diff].append(cat_questions[i])
        if all(len(tree[d]) >= top_k for d in tree):
            break

    return tree

if __name__ == "__main__":
    embeddings, metadata = load_embeddings()
    categories = sorted(set(metadata["categories"]))

    print("Available categories:")
    for i, cat in enumerate(categories):
        print(f"{i + 1}. {cat}")

    try:
        choice = int(input("\nSelect a category number: ")) - 1
        selected = categories[choice]

        tree = build_dynamic_tree(selected, embeddings, metadata)

        print(f"\n📘 RAG-based Interview Path for: {selected}")
        for level in ["Easy", "Medium", "Hard"]:
            if tree[level]:
                print(f"\n🔹 {level} Questions:")
                for q in tree[level]:
                    print(f"   • {q}")
            else:
                print(f"\n⚠️ No {level} questions found for {selected}")
    except (IndexError, ValueError):
        print("❌ Invalid input. Please select a valid category number.")

