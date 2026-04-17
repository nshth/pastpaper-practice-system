import json 
import os
import requests 
import chromadb
from dotenv import load_dotenv

load_dotenv(".env")

HF_API = os.getenv("HF_API")
MODEL = "sentence-transformers/all-mpnet-base-v2"

def get_embeddings(texts):
    response = requests.post(
        f"https://router.huggingface.co/hf-inference/models/{MODEL}/pipeline/feature-extraction",
        headers={"Authorization": f"Bearer {HF_API}"},
        json={"inputs": texts, "options": {"wait_for_model": True}}
    )
    return response.json()

units = json.load(open("units.json"))
docs = [u["unit_name"] + " " + u["combined_text"] + " " + " ".join(u["keywords"] * 5) for u in units]

embeddings = get_embeddings(docs)

client = chromadb.PersistentClient(path="./chroma_db")

try:
    client.delete_collection("units")
except:
    pass

collection = client.get_or_create_collection("units")
collection.add(
    ids=[u["unit_id"] for u in units],
    documents=docs,
    embeddings=embeddings,
    metadatas=[{"unit_name": u["unit_name"], "unit_number": int(u["unit_id"].split("_")[1])} for u in units]
)

print("Indexed", len(units), "units.")











