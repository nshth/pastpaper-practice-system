import os
import requests
import chromadb
from groq import Groq
from dotenv import load_dotenv
from .search_questions import get_unit_descriptions

load_dotenv(".env")

HF_API = os.getenv("HF_API")
EMBED_MODEL = "intfloat/e5-large-v2"
_groq = Groq(api_key=os.environ.get("GROQ_API"))
_chroma = chromadb.PersistentClient(path="./chroma_db")
_collection = _chroma.get_or_create_collection("units", metadata={"hnsw:space": "cosine"})


def _embed(text: str) -> list[float]:
    response = requests.post(
        f"https://router.huggingface.co/hf-inference/models/{EMBED_MODEL}/pipeline/feature-extraction",
        headers={"Authorization": f"Bearer {HF_API}"},
        json={"inputs": f"query: {text}", "options": {"wait_for_model": True}},
    )
    response.raise_for_status()
    return response.json()


def _retrieve_chunks(question_text: str, k: int = 3) -> list[str]:
    embedding = _embed(question_text)
    result = _collection.query(query_embeddings=[embedding], n_results=k, include=["documents"])
    return result["documents"][0]


def stream_explanation(question_text: str, correct_option: int | None = None):
    chunks = _retrieve_chunks(question_text)
    context = "\n\n".join(chunks)
    answer_line = f"The correct answer is option ({correct_option}).\n" if correct_option else ""

    prompt = f"""You are an ICT tutor helping a student understand a past exam question.

SYLLABUS CONTEXT:
{context}

QUESTION:
{question_text}

{answer_line}Explain why this is the correct answer using the syllabus context. Be clear and concise."""

    stream = _groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=600,
        stream=True,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token


def stream_general(user_message: str):
    unit_descriptions = get_unit_descriptions()

    prompt = f"""You are a helpful ICT study assistant for Grade 12/13 A-Level students.

SYLLABUS UNITS:
{unit_descriptions}

Answer the student's question with helpful, concise study guidance.

STUDENT: {user_message}"""

    stream = _groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=500,
        stream=True,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token