# embed_discourse.py
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_FILE = "discourse.index"
META_FILE = "discourse_meta.json"

# Load metadata
with open(META_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [item["text"] for item in data]

# Embed
model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save index
faiss.write_index(index, INDEX_FILE)
print(f"✅ Saved {len(texts)} Discourse embeddings to {INDEX_FILE}")
