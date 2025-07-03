#!/bin/bash

# Root folders
folders=(
"data/raw"
"data/processed"
"nlp"
"prompts"
"rag"
"multimodal"
"fine_tuning"
"evaluation"
"frontend"
"utils"
)

# Create folders and add .gitkeep
for folder in "${folders[@]}"; do
  mkdir -p "$folder"
  touch "$folder/.gitkeep"
done

# Add .gitignore if not present
cat <<EOL > .gitignore
__pycache__/
*.pyc
*.pkl
*.npy
.env
.DS_Store
*.pt
EOL

# Add README.md if not present
if [ ! -f "README.md" ]; then
  echo "# Mock Interview Assistant" > README.md
  echo "An AI-powered mock interview assistant using NLP, RAG, LLMs, LoRA, and multimodal feedback." >> README.md
fi

echo "✅ Folder structure created and ready."

