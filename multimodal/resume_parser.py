import spacy
import fitz  # PyMuPDF
import re

RESUME_PATH = "/Users/viswanathanrajavelu/mock-interview-assistant/resumes/Vignesh_Kumar_Rajavelu_Resume_1.pdf"  # <-- Update this path to your actual file

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_skills(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text.lower())

    # Very basic skill extraction: improve with custom patterns later
    skills = set()
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"}:
            skills.add(token.text)

    common_skills = {"python", "sql", "docker", "pytorch", "opencv", "fastapi", "tensorflow",
                     "linux", "git", "java", "nlp", "ml", "dl", "react", "kubernetes"}

    return sorted([skill for skill in skills if skill in common_skills])

if __name__ == "__main__":
    raw_text = extract_text_from_pdf(RESUME_PATH)
    skills = extract_skills(raw_text)
    print("\n🧠 Extracted Skills:")
    print(skills)

def extract_skills_from_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    return extract_skills(text)
