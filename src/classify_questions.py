import json
import os
import requests
import chromadb
import re
from dotenv import load_dotenv

load_dotenv(".env")

HF_API = os.getenv("HF_API")
MODEL = "sentence-transformers/all-mpnet-base-v2"

def get_embedding(text):
    response = requests.post(
        f"https://router.huggingface.co/hf-inference/models/{MODEL}/pipeline/feature-extraction",
        headers={"Authorization": f"Bearer {HF_API}"},
        json={"inputs": text, "options": {"wait_for_model": True}}
    )
    return response.json()

def clean_question(text):
    text = re.sub(r'\(1\).*', '', text, flags=re.DOTALL)
    return text.strip()


questions = json.load(open("data/extracted/merged/2024_P1_QA.json"))
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("units")

for q in questions:
    if "unit" in q:
        continue

    embedding = get_embedding(clean_question(q["text"]))
    result = collection.query(query_embeddings=[embedding], n_results=2)
    score1 = result["distances"][0][0]
    score2 = result["distances"][0][1]

    q["unit"] = result["metadatas"][0][0]["unit_number"]
    q["unit_confidence"] = round((2 - score1) / 2, 4)
    q["needs_review"] = (score2 - score1) < 0.05  # top-2 too close
    
json.dump(questions, open("questions.json", "w"), indent=2)
print("Done.")
