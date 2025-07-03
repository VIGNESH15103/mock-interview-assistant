import numpy as np
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

EMBEDDINGS_PATH = "data/processed/question_embeddings.npy"
METADATA_PATH = "data/processed/question_metadata.pkl"
OUTPUT_PLOT_PATH = "data/processed/tsne_plot.png"

# Load data
embeddings = np.load(EMBEDDINGS_PATH)
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

categories = metadata["categories"]
questions = metadata["questions"]

# Reduce dimensionality
print("🔄 Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
reduced = tsne.fit_transform(embeddings)

# Visualize
print("🎨 Generating scatter plot...")
plt.figure(figsize=(14, 10))
unique_labels = sorted(set(categories))
colors = plt.cm.get_cmap("tab20", len(unique_labels))

for idx, label in enumerate(unique_labels):
    points = reduced[np.array(categories) == label]
    plt.scatter(points[:, 0], points[:, 1], s=10, label=label, color=colors(idx))

plt.title("t-SNE Visualization of Interview Questions by Category")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(markerscale=2, fontsize=8, loc="best", ncol=2)
plt.tight_layout()

# Save the plot
plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
print(f"✅ t-SNE plot saved to: {OUTPUT_PLOT_PATH}")
