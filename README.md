# TDS Virtual TA 🤖📘

This is a Virtual Teaching Assistant (TA) for the **Tools in Data Science (TDS)** course (Jan–Apr 2025).  
It answers student questions based on course content and instructor replies on the TDS Discourse forum.

## 🔍 Features

- Accepts natural language questions from students
- Returns clear, concise answers (like a smart TA)
- Uses two knowledge bases:
  - 📘 TDS course materials (scraped from tds.s-anand.net)
  - 💬 Instructor/TA replies from Discourse (Jan–Apr 2025)
- Optionally supports image input (e.g. screenshots of questions)
- Provides source links to course/discourse pages
- Powered by [AI Pipe](https://aiproxy.s-anand.net) for generating answers

## 🚀 How it works

When you send a question (and optional image), the app:

1. Searches both course content and Discourse Q&A using FAISS
2. Selects the most relevant chunks based on cosine similarity
3. Builds a smart prompt with context and sends it to AI Pipe
4. Returns a summarized answer with supporting links

## 🧠 Model + Embeddings

- Embeddings: `all-MiniLM-L6-v2` (via Sentence Transformers)
- Answer generation: `gpt-4o-mini` via AI Pipe
- Vector search: FAISS (cosine similarity)

## 🛠 API Usage

### POST `/api/`

**Request JSON:**

```json
{
  "question": "How does the grading for GA4 bonus work?",
  "image": "optional base64 image string"
}


👩‍💻 Developer:

Developed by GARIMA

For the TDS Jan–Apr 2025 course project