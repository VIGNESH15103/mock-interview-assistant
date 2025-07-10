import pandas as pd
import re
import os

RAW_DATA_PATH = "data/raw/new_interview_questions.csv"
PROCESSED_DATA_PATH = "data/processed/cleaned_questions.csv"

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s\.,!?-]', '', text)
    return text.strip()

def preprocess():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)

    df["Question"] = df["Question"].astype(str).apply(clean_text)
    df["Answer"] = df["Answer"].astype(str).apply(clean_text)

    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"✅ Cleaned data saved to: {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    preprocess()
