from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import jieba
from rank_bm25 import BM25Okapi

from config.settings import SETTINGS
from utils.paths import ensure_dir


def _chunks_path(file_hash: str) -> Path:
    return SETTINGS.index_dir / f"{file_hash}.chunks.jsonl"


def _bm25_path(file_hash: str) -> Path:
    return SETTINGS.index_dir / f"{file_hash}.bm25.pkl"


def save_chunks_jsonl(chunks: List[dict], file_hash: str) -> Path:
    ensure_dir(SETTINGS.index_dir)
    path = _chunks_path(file_hash)
    if path.exists():
        path.unlink(missing_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for pos, c in enumerate(chunks):
            row = {
                "pos": pos,
                "chunk_id": c["chunk_id"],
                "parent_id": c["parent_id"],
                "text": c["text"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def load_chunks_jsonl(file_hash: str) -> List[dict]:
    path = _chunks_path(file_hash)
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    # pos 必须连续递增
    rows.sort(key=lambda x: int(x["pos"]))
    return rows


@dataclass
class BM25Index:
    file_hash: str
    bm25: BM25Okapi
    chunks: List[dict]  # pos aligned
    chunk_text_by_id: dict[str, str]

    def search(self, query: str, top_k: int) -> List[dict]:
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        # scores aligned with chunks pos
        idxs = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:top_k]
        out: list[dict] = []
        for i in idxs:
            out.append({"chunk_id": self.chunks[i]["chunk_id"], "score": float(scores[i])})
        return out


def build_index(chunks: List[dict], file_hash: str, k1: float, b: float) -> Path:
    ensure_dir(SETTINGS.index_dir)
    path = _bm25_path(file_hash)
    if path.exists():
        path.unlink(missing_ok=True)

    docs = [c["text"] for c in chunks]
    tokenized_docs = [list(jieba.cut(d)) for d in docs]
    bm25 = BM25Okapi(tokenized_docs, k1=k1, b=b)
    payload = {
        "k1": k1,
        "b": b,
        "tokenized_docs": tokenized_docs,
        "bm25": bm25,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


def load_index(file_hash: str) -> BM25Index:
    with open(_bm25_path(file_hash), "rb") as f:
        payload = pickle.load(f)
    chunks = load_chunks_jsonl(file_hash)
    bm25: BM25Okapi = payload["bm25"]
    chunk_text_by_id = {c["chunk_id"]: c["text"] for c in chunks}
    return BM25Index(file_hash=file_hash, bm25=bm25, chunks=chunks, chunk_text_by_id=chunk_text_by_id)

