import json

# Load original Q&A pairs
with open("qa_pairs.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Prepare the discourse_meta format
discourse_meta = []
for item in data:
    question = item["question"].strip()
    answer = item["answer"].strip()
    url = item["url"].strip()
    
    # Combine Q and A for embedding
    full_text = f"Q: {question}\nA: {answer}"

    discourse_meta.append({
        "text": full_text,
        "original_url": url
    })

# Save to new metadata file
with open("discourse_meta.json", "w", encoding="utf-8") as f:
    json.dump(discourse_meta, f, indent=2, ensure_ascii=False)

print("✅ discourse_meta.json created successfully with combined Q+A.")
