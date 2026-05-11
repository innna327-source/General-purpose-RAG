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
    parent_text_by_id: dict[str, str]  # parent_id -> 合并后的完整父文档文本
    chunk_to_parent: dict[str, str]  # chunk_id -> parent_id 映射
    chunks_by_pos: List[dict]  # 按 pos 排序的所有 chunks

    def search(self, query: str, top_k: int) -> List[dict]:
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        # scores aligned with chunks pos
        idxs = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:top_k]
        out: list[dict] = []
        for i in idxs:
            out.append({"chunk_id": self.chunks[i]["chunk_id"], "score": float(scores[i]), "pos": i})
        return out

    def get_parent_text(self, chunk_id: str) -> str:
        """根据chunk_id获取其所属父文档的完整文本"""
        parent_id = self.chunk_to_parent.get(chunk_id)
        if parent_id:
            return self.parent_text_by_id.get(parent_id, "")
        return ""

    def expand_chunk_window(self, chunk_id: str, window_size: int = 2) -> str:
        """
        窗口扩展：返回命中chunk及其前后相邻chunk的合并文本

        参数：
        - chunk_id: 命中的chunk
        - window_size: 前后扩展多少个chunk（默认前后各2个）

        返回：合并后的扩展文本
        """
        # 找到该chunk的pos位置
        chunk_pos = None
        for i, c in enumerate(self.chunks_by_pos):
            if c["chunk_id"] == chunk_id:
                chunk_pos = i
                break

        if chunk_pos is None:
            return self.chunk_text_by_id.get(chunk_id, "")

        # 计算窗口范围
        start = max(0, chunk_pos - window_size)
        end = min(len(self.chunks_by_pos), chunk_pos + window_size + 1)

        # 只合并同一个父文档内的chunk（避免跨主题）
        target_parent = self.chunks_by_pos[chunk_pos].get("parent_id", "")
        merged_texts = []
        for i in range(start, end):
            c = self.chunks_by_pos[i]
            if c.get("parent_id") == target_parent:
                merged_texts.append(c.get("text", ""))

        return "\n".join(merged_texts)


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

    # 按pos排序保存（用于窗口扩展）
    chunks_by_pos = sorted(chunks, key=lambda x: int(x.get("pos", 0)))

    # 构建父文档映射：parent_id -> 合并后的完整文本
    parent_chunks: dict[str, list] = {}
    chunk_to_parent: dict[str, str] = {}
    for c in chunks:
        parent_id = c.get("parent_id", "")
        chunk_id = c.get("chunk_id", "")
        if parent_id:
            if parent_id not in parent_chunks:
                parent_chunks[parent_id] = []
            parent_chunks[parent_id].append(c)
            chunk_to_parent[chunk_id] = parent_id

    # 按pos排序后合并每个父文档的所有子chunk文本
    parent_text_by_id: dict[str, str] = {}
    for parent_id, children in parent_chunks.items():
        children_sorted = sorted(children, key=lambda x: int(x.get("pos", 0)))
        merged_text = "\n".join([child.get("text", "") for child in children_sorted])
        parent_text_by_id[parent_id] = merged_text

    return BM25Index(
        file_hash=file_hash,
        bm25=bm25,
        chunks=chunks,
        chunk_text_by_id=chunk_text_by_id,
        parent_text_by_id=parent_text_by_id,
        chunk_to_parent=chunk_to_parent,
        chunks_by_pos=chunks_by_pos,
    )

