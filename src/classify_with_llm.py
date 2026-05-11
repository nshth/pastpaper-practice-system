"""
LLM fallback classifier for questions flagged needs_review=True.

Uses the top-K chunk texts already stored in questions.json (by classify_questions.py)
to give the LLM actual syllabus context — not just unit names.

Model rotation cycles through Groq models to stay within rate limits.
"""

import json
import os
import re
import itertools
from dotenv import load_dotenv
from groq import Groq

load_dotenv(".env")

client = Groq(api_key=os.environ.get("GROQ_API"))

MODELS = itertools.cycle([
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
])

UNIT_NAMES = {
    1:  "Concept of ICT",
    2:  "Introduction to Computer",
    3:  "Data Representation",
    4:  "Fundamental of Digital Circuits",
    5:  "Computer Operating System",
    6:  "Data Communication and Networking",
    7:  "System Analysis and Design",
    8:  "Database Management",
    9:  "Programming",
    10: "Web Development",
    11: "Internet of Things",
    12: "ICT in Business",
    13: "New Trends and Future Directions of ICT",
    14: "Project",
}

UNITS_LIST = "\n".join(f"{k}. {v}" for k, v in UNIT_NAMES.items())


def build_prompt(question_text: str, top_docs: list[dict]) -> str:
    context_blocks = ""
    for doc in top_docs:
        context_blocks += (
            f"\n--- Unit {doc['unit_number']}: {doc['unit_name']} ---\n"
            f"{doc['text']}\n"
        )

    return f"""You are classifying a Grade 12, 13 ICT exam question into one of 14 syllabus units.

ALL UNITS:
{UNITS_LIST}

CANDIDATE UNIT CONTEXT (retrieved from the syllabus):
{context_blocks}

QUESTION:
{question_text}

Based on the question and the syllabus context above, which unit does this question belong to?
Reply with ONLY the unit number (1-14). No explanation."""


def extract_unit_number(response_text: str) -> int | None:
    m = re.search(r'\b(\d{1,2})\b', response_text.strip())
    if m:
        n = int(m.group(1))
        if 1 <= n <= 14:
            return n
    return None


def classify_with_llm(question_text: str, top_docs: list[dict], model: str) -> int | None:
    prompt = build_prompt(question_text, top_docs)
    result = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_completion_tokens=10,
    )
    raw = result.choices[0].message.content.strip()
    return extract_unit_number(raw)


questions = json.load(open("questions.json"))
needs_llm = [q for q in questions if q.get("needs_review") and q.get("unit_source") != "llm"]
print(f"{len(needs_llm)} questions need LLM classification.")

for q in needs_llm:
    model = next(MODELS)
    top_docs = q.get("_top_docs_for_llm", [])

    try:
        unit_num = classify_with_llm(q["text"], top_docs, model)
        if unit_num is None:
            raise ValueError("Could not parse unit number from response")

        q["unit"] = unit_num
        q["unit_source"] = "llm"
        q["llm_model"] = model
        q["needs_review"] = False
        print(f"{q.get('question_id', '?')} -> unit {unit_num} ({model})")

    except Exception as e:
        print(f"{q.get('question_id', '?')} FAILED ({model}): {e}")
        try:
            model = next(MODELS)
            unit_num = classify_with_llm(q["text"], top_docs, model)
            if unit_num:
                q["unit"] = unit_num
                q["unit_source"] = "llm_retry"
                q["llm_model"] = model
                q["needs_review"] = False
                print(f"  retry -> unit {unit_num} ({model})")
        except Exception as e2:
            print(f"  retry also failed: {e2}")

# Clean up internal field before saving
for q in questions:
    q.pop("_top_docs_for_llm", None)

json.dump(questions, open("questions.json", "w"), indent=2)
print("Done.")