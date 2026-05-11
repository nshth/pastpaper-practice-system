import json
import math
import os
import re
import requests
import chromadb
from collections import Counter, defaultdict
from dotenv import load_dotenv

load_dotenv(".env")

HF_API = os.getenv("HF_API")
EMBED_MODEL = "intfloat/e5-large-v2"
TOP_K = 5
CONFIDENCE_THRESHOLD = 0.40   # Lowered from 0.50 to reduce LLM overhead
MARGIN_THRESHOLD = 0.02

def get_query_embedding(text: str) -> list[float]:
    response = requests.post(
        f"https://router.huggingface.co/hf-inference/models/{EMBED_MODEL}/pipeline/feature-extraction",
        headers={"Authorization": f"Bearer {HF_API}"},
        json={"inputs": f"query: {text}", "options": {"wait_for_model": True}},
    )
    response.raise_for_status()
    return response.json()

def bm25_score(query_terms: list[str], doc: dict, avgdl: float, idf: dict, k1: float, b: float) -> float:
    score = 0.0
    dl = doc["dl"]
    tf_map = doc["tf"]
    for term in query_terms:
        if term not in tf_map: continue
        tf = tf_map[term]
        score += idf.get(term, 0) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
    return score

STOP = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "in", "on", "at", "to", "for", "with", "by", "from", "and", "or", "but", "not", "what", "which", "how", "that", "this", "it", "its", "as", "if", "will", "can", "do", "does"}

def preprocess_query(text: str) -> str:
    text = re.sub(r'\(1\).*', '', text, flags=re.DOTALL)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize(text: str) -> list[str]:
    tokens = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
    return [t for t in tokens if t not in STOP]

def rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def compute_confidence(dense_distances: list[float], bm25_scores_top: list[float], query_tokens: list[str], chunk_keywords: list[list[str]]) -> tuple[float, float]:
    # 1. Dense Signal (Cosine Sim)
    d1 = max(0.0, min(1.0, dense_distances[0])) if dense_distances else 1.0
    d2 = max(0.0, min(1.0, dense_distances[1])) if len(dense_distances) > 1 else 1.0
    dense_sim1 = 1.0 - d1
    dense_margin = dense_sim1 - (1.0 - d2)

    # 2. BM25 Signal - FIXED: Added epsilon check to prevent exploding scores
    b1, b2 = max(0.0, bm25_scores_top[0]), (max(0.0, bm25_scores_top[1]) if len(bm25_scores_top) > 1 else 0.0)
    total = b1 + b2
    bm25_signal = (b1 / total) if total > 0.01 else 0.5
    bm25_norm = (bm25_signal - 0.5) * 2

    # 3. Keyword Match
    query_set, kw_set = set(query_tokens), set(chunk_keywords[0]) if chunk_keywords else set()
    overlap = len(query_set & kw_set) / max(len(query_set), 1)
    kw_score = min(overlap * 2, 1.0)

    # Weighted Confidence: 50% Dense, 30% BM25, 20% Keyword
    confidence = 0.5 * dense_sim1 + 0.3 * bm25_norm + 0.2 * kw_score
    return round(confidence, 4), round(dense_margin, 4)

# Load Indexes
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("units", metadata={"hnsw:space": "cosine"})
bm25_data = json.load(open("bm25_index.json"))
BM25_K1, BM25_B, BM25_AVGDL, BM25_IDF, BM25_DOCS, BM25_IDS = bm25_data["k1"], bm25_data["b"], bm25_data["avgdl"], bm25_data["idf"], bm25_data["docs"], bm25_data["chunk_ids"]
BM25_ID_TO_IDX = {cid: i for i, cid in enumerate(BM25_IDS)}
units_chunks = json.load(open("units_chunks.json"))
chunk_keywords_by_id = {c["unit_id"] + "_" + (c["competency_level"] or "fallback"): c["keywords"] for c in units_chunks}

# Main Loop
questions = json.load(open("data/extracted/merged/2024_P1_QA.json"))
for q in questions:
    if "unit" in q and not q.get("needs_review"): continue
    raw_query = preprocess_query(q["text"])
    query_tokens = tokenize(raw_query)

    query_emb = get_query_embedding(raw_query)
    dense_result = collection.query(query_embeddings=[query_emb], n_results=TOP_K, include=["distances", "metadatas", "documents"])
    dense_ids, dense_distances, dense_docs, dense_metas = dense_result["ids"][0], dense_result["distances"][0], dense_result["documents"][0], dense_result["metadatas"][0]

    bm25_scores_raw = [(BM25_IDS[i], bm25_score(query_tokens, BM25_DOCS[i], BM25_AVGDL, BM25_IDF, BM25_K1, BM25_B)) for i in range(len(BM25_DOCS))]
    bm25_ranked = [doc_id for doc_id, _ in sorted(bm25_scores_raw, key=lambda x: x[1], reverse=True)][:TOP_K]

    fused = rrf_fuse([dense_ids, bm25_ranked])
    top_fused_ids = [doc_id for doc_id, _ in fused[:TOP_K]]

    dense_meta_by_id = {dense_ids[i]: dense_metas[i] for i in range(len(dense_ids))}
    dense_dist_by_id = {dense_ids[i]: dense_distances[i] for i in range(len(dense_ids))}
    dense_doc_by_id  = {dense_ids[i]: dense_docs[i] for i in range(len(dense_ids))}

    for mid in [i for i in top_fused_ids if i not in dense_meta_by_id]:
        extra = collection.get(ids=[mid], include=["metadatas", "documents"])
        dense_meta_by_id[mid], dense_doc_by_id[mid], dense_dist_by_id[mid] = extra["metadatas"][0], extra["documents"][0], 2.0

    combined = sorted(zip(top_fused_ids, [dense_meta_by_id[i] for i in top_fused_ids], [dense_dist_by_id[i] for i in top_fused_ids], [dense_doc_by_id[i] for i in top_fused_ids]), key=lambda x: x[2])
    top_ids, top_metas, top_dists, top_docs = map(list, zip(*combined))

    top_bm25_scores = [bm25_score(query_tokens, BM25_DOCS[BM25_ID_TO_IDX[cid]], BM25_AVGDL, BM25_IDF, BM25_K1, BM25_B) if cid in BM25_ID_TO_IDX else 0.0 for cid in top_ids[:2]]
    confidence, margin = compute_confidence(top_dists, top_bm25_scores, query_tokens, [chunk_keywords_by_id.get(cid, []) for cid in top_ids])

    q["unit"], q["unit_confidence"], q["unit_margin"], q["needs_review"], q["unit_source"] = top_metas[0]["unit_number"], confidence, margin, (confidence < CONFIDENCE_THRESHOLD), "retrieval"
    q["top_units"], q["_top_docs_for_llm"] = [m["unit_number"] for m in top_metas[:3]], [{"unit_number": top_metas[i]["unit_number"], "unit_name": top_metas[i]["unit_name"], "text": top_docs[i][:600]} for i in range(min(3, len(top_metas)))]
    print(f"{q.get('question_id','?')} -> unit {q['unit']} | conf={confidence} | {'LOW -> LLM' if q['needs_review'] else 'OK'}")

json.dump(questions, open("questions.json", "w"), indent=2)