import json
import os
import math
import requests
import chromadb
from collections import Counter
from dotenv import load_dotenv

load_dotenv(".env")

HF_API = os.getenv("HF_API")
EMBED_MODEL = "intfloat/e5-large-v2"

def get_embeddings(texts: list[str]) -> list[list[float]]:
    prefixed = [f"passage: {t}" for t in texts]
    response = requests.post(
        f"https://router.huggingface.co/hf-inference/models/{EMBED_MODEL}/pipeline/feature-extraction",
        headers={"Authorization": f"Bearer {HF_API}"},
        json={"inputs": prefixed, "options": {"wait_for_model": True}},
    )
    response.raise_for_status()
    return response.json()

def build_bm25_index(chunks: list[dict]) -> dict:
    k1, b = 1.5, 0.75
    tokenize = lambda t: t.lower().split()

    corpus = [tokenize(c["text"] + " " + " ".join(c["keywords"])) for c in chunks]
    avgdl = sum(len(d) for d in corpus) / len(corpus)
    N = len(corpus)

    df: Counter = Counter()
    for doc in corpus:
        for term in set(doc):
            df[term] += 1

    idf = {
        term: math.log((N - freq + 0.5) / (freq + 0.5) + 1)
        for term, freq in df.items()
    }

    doc_term_freqs = []
    for doc in corpus:
        tf = Counter(doc)
        dl = len(doc)
        doc_term_freqs.append({"tf": dict(tf), "dl": dl})

    return {
        "k1": k1, "b": b, "avgdl": avgdl, "idf": idf, "docs": doc_term_freqs,
        "chunk_ids": [c["unit_id"] + "_" + (c["competency_level"] or "fallback") for c in chunks],
    }

chunks: list[dict] = json.load(open("units_chunks.json"))

docs = []
for c in chunks:
    comp_prefix = f"Competency Level {c['competency_level']} " if c["competency_level"] else ""
    kw_boost = " ".join(c["keywords"] * 3)
    docs.append(f"{c['unit_name']} {comp_prefix}{c['text']} {kw_boost}")

print(f"Embedding {len(docs)} chunks...")
BATCH = 8
all_embeddings = []
for i in range(0, len(docs), BATCH):
    batch = docs[i:i + BATCH]
    embs = get_embeddings(batch)
    all_embeddings.extend(embs)
    print(f"  {min(i + BATCH, len(docs))}/{len(docs)}")

client = chromadb.PersistentClient(path="./chroma_db")
try:
    client.delete_collection("units")
except Exception:
    pass

collection = client.get_or_create_collection("units", metadata={"hnsw:space": "cosine"})
collection.add(
    ids=[c["unit_id"] + "_" + (c["competency_level"] or "fallback") for c in chunks],
    documents=docs,
    embeddings=all_embeddings,
    metadatas=[
        {
            "unit_id": c["unit_id"],
            "unit_number": c["unit_number"],
            "unit_name": c["unit_name"],
            "competency_level": c["competency_level"] or "",
        }
        for c in chunks
    ],
)
print(f"ChromaDB: indexed {len(chunks)} chunks.")

bm25 = build_bm25_index(chunks)
json.dump(bm25, open("bm25_index.json", "w"))
print("BM25 index saved.")