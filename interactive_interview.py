from multimodal.resume_parser import extract_skills_from_pdf
from rag.retriever import retrieve_similar_questions
import pickle, numpy as np
from transformers import pipeline

# Paths
EMBEDDINGS_PATH = "data/processed/question_embeddings.npy"
METADATA_PATH = "data/processed/question_metadata.pkl"
RESUME_PATH = "resumes/Vignesh_Kumar_Rajavelu_Resume_1.pdf"
TOP_K = 3

# 🔧 GPT2 question generator
generator = pipeline("text-generation", model="gpt2")

def generate_questions(skill, difficulty="Easy"):
    prompt = f"Generate a {difficulty} technical interview question about {skill}."
    response = generator(prompt, max_length=50, num_return_sequences=1)
    return response[0]['generated_text']

def load_embeddings():
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return embeddings, metadata

def ask_questions_loop(skills, embeddings, metadata):
    for skill in skills:
        print(f"\n💡 Starting questions for skill: {skill.upper()}")
        tree = retrieve_similar_questions(skill)
        difficulty_levels = ["Easy", "Medium", "Hard"]
        current_level = 0  # Start at Easy

        while 0 <= current_level < len(difficulty_levels):
            level = difficulty_levels[current_level]
            questions = tree[level]

            if questions:
                question = questions.pop(0)
            else:
                print(f"⚠️ No {level} questions found for {skill}. Generating one...")
                question = generate_questions(skill, level)

            print(f"\n🧠 [{level}] {question}")
            answer = input("Your answer: ")

            if len(answer.strip()) > 15:
                print("✅ Looks good!")
                current_level += 1
            else:
                print("❌ That was weak.")
                current_level -= 1

def main():
    skills = extract_skills_from_pdf(RESUME_PATH)
    print(f"\n📄 Extracted Skills:\n{skills}")
    embeddings, metadata = load_embeddings()
    ask_questions_loop(skills, embeddings, metadata)

if __name__ == "__main__":
    main()
