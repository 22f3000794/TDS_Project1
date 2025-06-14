import os
import json
import faiss
import numpy as np
import requests
from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Optional
from fastapi.responses import JSONResponse

# ---------------------- Configuration ----------------------
COURSE_INDEX_FILE = "course_content.index"
COURSE_META_FILE = "course_content_meta.json"
DISCOURSE_INDEX_FILE = "discourse.index"
DISCOURSE_META_FILE = "discourse_meta.json"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
TOP_K = 5
SIM_THRESHOLD = 0.45

# AI Pipe config
API_KEY = os.getenv("AI_PIPE_API_KEY")
#AI_PIPE_API_KEY = "YOUR_AIPIPE_API_KEY"
AI_PIPE_URL = "https://aipipe.org/openrouter/v1/chat/completions"
AI_PIPE_MODEL = "openai/gpt-4.1-nano"

# ---------------------- Load model and indexes ----------------------
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

course_index = faiss.read_index(COURSE_INDEX_FILE)
with open(COURSE_META_FILE, "r", encoding="utf-8") as f:
    course_meta = json.load(f)

discourse_index = faiss.read_index(DISCOURSE_INDEX_FILE)
with open(DISCOURSE_META_FILE, "r", encoding="utf-8") as f:
    discourse_meta = json.load(f)

# ---------------------- FastAPI setup ----------------------
app = FastAPI()

# ---------------------- Request & Response Schemas ----------------------
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # currently unused

class Link(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[Link]

# ---------------------- Utility Functions ----------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def generate_answer_with_aipipe(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": AI_PIPE_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(AI_PIPE_URL, headers=headers, json=payload, timeout=25)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"❌ Error generating answer from AI Pipe: {e}"

# ---------------------- API Endpoint ----------------------
@app.post("/api/", response_model=QueryResponse)
async def answer_question(req: QueryRequest):
    query = req.question.strip()
    query_embedding = model.encode([query]).astype("float32")

    # Search course
    D_course, I_course = course_index.search(query_embedding, TOP_K)
    best_course = max(
        ((int(I_course[0][i]), cosine_similarity(query_embedding[0], course_index.reconstruct(int(I_course[0][i]))))
         for i in range(TOP_K) if I_course[0][i] != -1),
        key=lambda x: x[1],
        default=(-1, -1)
    )

    # Search discourse
    D_discourse, I_discourse = discourse_index.search(query_embedding, TOP_K)
    best_disc = max(
        ((int(I_discourse[0][i]), cosine_similarity(query_embedding[0], discourse_index.reconstruct(int(I_discourse[0][i]))))
         for i in range(TOP_K) if I_discourse[0][i] != -1),
        key=lambda x: x[1],
        default=(-1, -1)
    )

    # Choose best
    context, url, text = "", "", ""
    if best_course[1] >= best_disc[1] and best_course[1] > SIM_THRESHOLD:
        result = course_meta[best_course[0]]
        context = result["text"].strip()
        url = result["original_url"]
        text = "Answer from course content"
    elif best_disc[1] > SIM_THRESHOLD:
        result = discourse_meta[best_disc[0]]
        context = result["text"].split("A:", 1)[-1].strip()
        url = result["original_url"]
        text = "Answer from Discourse"
    else:
        return JSONResponse(status_code=200, content={"answer": "❌ Sorry, I couldn't find a relevant answer.", "links": []})

    # Generate answer
    full_prompt = f"""
    You are a helpful virtual TA for the Tools in Data Science (TDS) course. 
    Using only the information in the context below, answer the student question clearly, concisely, and in your own words. 
    If the context includes a response from a TA or instructor, summarize it accurately but naturally. 
    Avoid guessing if the answer is not clearly present.

    Context:
    {context}

    Question: {query}

    Answer:
    """.strip()
    answer = generate_answer_with_aipipe(full_prompt)

    return {
        "answer": answer,
        "links": [{"url": url, "text": text}]
    }

