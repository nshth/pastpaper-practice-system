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


def _build_history_messages(history: list[dict]) -> list[dict]:
    recent = [h for h in history if h["role"] in ("user", "assistant")][-6:]
    return [{"role": h["role"], "content": h["content"]} for h in recent]


def stream_explanation(question_text: str, correct_option, history: list[dict]):
    if isinstance(correct_option, list):
        correct_option = correct_option[0] if correct_option else None

    chunks = _retrieve_chunks(question_text)
    context = "\n\n".join(chunks)
    answer_line = f"The correct answer is option ({correct_option}).\n" if correct_option else ""

    system = f"""You are an ICT tutor helping a Grade 12/13 student understand past exam questions.
Use the syllabus context below to explain clearly and concisely.

SYLLABUS CONTEXT:
{context}"""

    messages = [{"role": "system", "content": system}]
    messages += _build_history_messages(history)
    messages.append({
        "role": "user",
        "content": f"Explain this question:\n\n{question_text}\n\n{answer_line}"
    })

    stream = _groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,
        max_tokens=600,
        stream=True,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token


def stream_general(user_message: str, history: list[dict]):
    unit_descriptions = get_unit_descriptions()

    system = f"""You are a helpful ICT study assistant for Grade 12/13 A-Level students.

SYLLABUS UNITS:
{unit_descriptions}

Give helpful, concise study guidance."""

    messages = [{"role": "system", "content": system}]
    messages += _build_history_messages(history)
    messages.append({"role": "user", "content": user_message})

    stream = _groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.5,
        max_tokens=500,
        stream=True,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token