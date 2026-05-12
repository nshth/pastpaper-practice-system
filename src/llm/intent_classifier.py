import json
import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv(".env")

_client = Groq(api_key=os.environ.get("GROQ_API"))

SYSTEM = """You are an intent classifier for a past paper practice system.
Classify the user message into one of: search, explain, general.

- search: user wants to find/list questions (by unit, topic, year)
- explain: user wants explanation of a question or answer or concept
- general: study advice, tips, anything else

Reply ONLY with valid JSON, no markdown:
{"intent": "search", "unit": <int or null>, "year": <int or null>}
{"intent": "explain", "question_text": "<paste or summary>"}
{"intent": "general"}"""


def detect_intent(user_message: str) -> dict:
    response = _client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        max_tokens=80,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"intent": "general"}