import random
import pandas as pd

DATA_PATH = "data/raw/new_interview_questions.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def build_tree(df):
    tree = {}
    for category in df["Category"].unique():
        cat_df = df[df["Category"] == category]
        tree[category] = {
            "Easy": cat_df[cat_df["Difficulty"] == "Easy"]["Question"].tolist(),
            "Medium": cat_df[cat_df["Difficulty"] == "Medium"]["Question"].tolist(),
            "Hard": cat_df[cat_df["Difficulty"] == "Hard"]["Question"].tolist()
        }
    return tree

def simulate_interview(tree, category):
    print(f"\n📘 Interview Path for: {category}")
    for level in ["Easy", "Medium", "Hard"]:
        questions = tree[category].get(level, [])
        if questions:
            q = random.choice(questions)
            print(f"🔹 {level}: {q}")
        else:
            print(f"⚠️ No questions found for {level} in {category}")

if __name__ == "__main__":
    df = load_data()
    tree = build_tree(df)

    available_categories = list(tree.keys())
    print("Available categories:")
    for i, cat in enumerate(available_categories):
        print(f"{i + 1}. {cat}")

    choice = int(input("\nSelect a category number: ")) - 1
    selected = available_categories[choice]
    simulate_interview(tree, selected)
