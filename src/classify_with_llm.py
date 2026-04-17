import json
import os
import re
import itertools
from dotenv import load_dotenv
from groq import Groq

load_dotenv(".env")

client = Groq(
    api_key=os.environ.get("GROQ_API"))

MODELS = itertools.cycle([
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
])

units = json.load(open("units.json"))

units_summary = "\n".join(
    f"{u['unit_id'].split('_')[1]}. {u['unit_name']}: {u['description']}"
    for u in units
)

def classify_with_llm(question_text, model):
    prompt = f"""You are classifying ICT exam questions into units.
        UNITS:
        {units_summary}

        QUESTION:
        {question_text}

        Reply with ONLY the unit number (1-14). Nothing else."""

    result = client.chat.completions.create(
        model=model,
        messages=[
        {
            "role": "user",
            "content": prompt
        }
        ],
        temperature=1,
        max_completion_tokens=10
    )

    return result.choices[0].message.content.strip()

questions = json.load(open("questions.json"))

for q in questions:
    if not q.get("needs_review") and q.get("unit"):
        continue

    model = next(MODELS)
    try:
        unit_num = classify_with_llm(q["text"], model)
        q["unit"] = unit_num,
        q["unit_source"] = "llm"
        q["llm_model"] = model
        print(f"{q['question_id']} -> unit {unit_num} ({model})")

    except Exception as e:
        print(f"{q['question_id']} FAILED ({model}): {e}")
        # rotate to next model on failure
        model = next(MODELS)
        unit_num = classify_with_llm(q["text"], model)
        q["unit"] = unit_num
        q["unit_source"] = "llm"
        q["llm_model"] = model

json.dump(questions, open("questions.json", "w"), indent=2)
print("Done.")
