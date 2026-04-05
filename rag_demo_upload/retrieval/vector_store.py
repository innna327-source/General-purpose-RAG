from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import SETTINGS
from retrieval.bm25 import load_chunks_jsonl
from utils.paths import ensure_dir


def _faiss_path(file_hash: str) -> Path:
    return SETTINGS.index_dir / f"{file_hash}.faiss"


@dataclass
class VectorStore:
    file_hash: str
    index: object
    chunks: List[dict]  # pos aligned
    chunk_text_by_id: dict[str, str]
    embedder: SentenceTransformer

    def search(self, query: str, top_k: int) -> List[dict]:
        q = self.embedder.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        q = np.ascontiguousarray(q)
        D, I = self.index.search(q, top_k)
        out: list[dict] = []
        for score, pos in zip(D[0].tolist(), I[0].tolist()):
            if pos is None or int(pos) < 0:
                continue
            out.append({"chunk_id": self.chunks[int(pos)]["chunk_id"], "score": float(score)})
        return out


def build_index(chunks: List[dict], file_hash: str, model_name: str) -> Path:
    ensure_dir(SETTINGS.index_dir)
    path = _faiss_path(file_hash)
    if path.exists():
        raise ValueError(f"FAISS 索引已存在，禁止覆盖：{path}")

    try:
        import faiss
    except Exception as e:
        raise ImportError(
            "未检测到 faiss。Windows 推荐：conda install -c conda-forge faiss-cpu"
        ) from e

    embedder = SentenceTransformer(model_name, device="cpu")
    texts = [c["text"] for c in chunks]
    X = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    X = np.ascontiguousarray(X)
    dim = X.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(X)
    faiss.write_index(index, str(path))
    return path


def load_index(file_hash: str, model_name: str) -> VectorStore:
    try:
        import faiss
    except Exception as e:
        raise ImportError(
            "未检测到 faiss。Windows 推荐：conda install -c conda-forge faiss-cpu"
        ) from e

    idx = faiss.read_index(str(_faiss_path(file_hash)))
    chunks = load_chunks_jsonl(file_hash)
    chunk_text_by_id = {c["chunk_id"]: c["text"] for c in chunks}
    embedder = SentenceTransformer(model_name, device="cpu")
    return VectorStore(
        file_hash=file_hash, index=idx, chunks=chunks, chunk_text_by_id=chunk_text_by_id, embedder=embedder
    )

