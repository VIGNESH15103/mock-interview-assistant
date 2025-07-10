import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from multimodal.resume_parser import extract_skills_from_pdf

EMBEDDINGS_PATH = "data/processed/question_embeddings.npy"
METADATA_PATH = "data/processed/question_metadata.pkl"
INDEX_PATH = "data/processed/faiss_index.index"
RESUME_PATH = "resumes/Vignesh_Kumar_Rajavelu_Resume_1.pdf"

TOP_K = 3

def load_embeddings_and_metadata():
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return embeddings, metadata

def retrieve_similar_questions(skill, embeddings, metadata, top_k=3):
    questions = metadata["questions"]
    categories = metadata["categories"]
    difficulty = metadata["difficulty"]

    # Simulate skill embedding as a centroid-style pseudo-query
    skill_embedding = np.mean(embeddings, axis=0, keepdims=True)
    sim_scores = cosine_similarity(embeddings, skill_embedding).flatten()

    ranked_indices = np.argsort(sim_scores)[::-1]
    
    results = {"Easy": [], "Medium": [], "Hard": []}
    for idx in ranked_indices:
        if skill.lower() in questions[idx].lower():
            diff = difficulty[idx]
            if diff in results and len(results[diff]) < top_k:
                results[diff].append(questions[idx])
            if all(len(results[d]) == top_k for d in results):
                break
    return results

def display_interview_tree(skill, tree):
    print(f"\n📘 Interview Path for: {skill}")
    for level in ["Easy", "Medium", "Hard"]:
        if tree[level]:
            print(f"\n🔹 {level} Questions:")
            for q in tree[level]:
                print(f"   • {q}")
        else:
            print(f"\n⚠️ No {level} questions found.")

if __name__ == "__main__":
    # Extract skills
    print("📄 Parsing resume and extracting skills...")
    skills = extract_skills_from_pdf(RESUME_PATH)
    print(f"🧠 Extracted Skills:\n{skills}")

    embeddings, metadata = load_embeddings_and_metadata()

    for skill in skills:
        tree = retrieve_similar_questions(skill, embeddings, metadata, top_k=TOP_K)
        display_interview_tree(skill, tree)
