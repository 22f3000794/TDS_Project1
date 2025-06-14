import os
import re
import glob
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from datetime import datetime

# CONFIG
MD_DIR = "markdown_files"
INDEX_FILE = "course_content.index"
META_FILE = "course_content_meta.json"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def clean_ui_artifacts(text):
    # Remove all image markdown (e.g., YouTube thumbnails)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

    # Remove empty or malformed markdown links
    text = re.sub(r'\[\]\([^\)]*\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\(\#?\)', r'\1', text)

    # Remove known UI junk
    text = text.replace("Copy to clipboard", "").replace("ErrorCopied", "").replace("Code snippet", "")

    # Normalize excessive line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

def extract_links(text):
    # Return list of all clickable links (YouTube, GitHub, etc.)
    return re.findall(r'https:\/\/[^\s\)]+', text)

def extract_frontmatter(md_text):
    match = re.match(r'^---\n(.*?)\n---\n(.*)', md_text, flags=re.DOTALL)
    if match:
        frontmatter_raw, content = match.groups()
        frontmatter = {}
        for line in frontmatter_raw.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                frontmatter[key.strip()] = value.strip().strip('"')
        return frontmatter, content
    else:
        return {}, md_text

def chunk_text(text, max_len=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_len, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
    return chunks

# Embedding logic
all_chunks = []
metadata = []

for filepath in glob.glob(os.path.join(MD_DIR, "*.md")):
    with open(filepath, "r", encoding="utf-8") as f:
        md_text = f.read()

    frontmatter, content = extract_frontmatter(md_text)
    cleaned = clean_ui_artifacts(content)
    chunks = chunk_text(cleaned)

    for i, chunk in enumerate(chunks):
        links = extract_links(chunk)
        all_chunks.append(chunk)
        metadata.append({
            "source_file": os.path.basename(filepath),
            "text": chunk,
            "original_url": frontmatter.get("original_url", "URL not available"),
            "extra_links": links
        })

print(f"🔍 Embedding {len(all_chunks)} chunks...")

embeddings = model.encode(all_chunks, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, INDEX_FILE)
with open(META_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Done. Index: {INDEX_FILE}, Metadata: {META_FILE}")
