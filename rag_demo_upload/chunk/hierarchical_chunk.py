from __future__ import annotations

import re
from typing import Dict, List, Optional

from transformers import AutoTokenizer

from chunk.base_chunker import BaseChunker, ChunkResult
from utils.hash_utils import sha256_text


# 默认标题正则，覆盖通用中文章节结构，可在构造时传入自定义模式覆盖
_DEFAULT_TITLE_PATTERNS: List[re.Pattern] = [
    re.compile(r"^\s*#"),
    re.compile(r"^\s*第\s*[一二三四五六七八九十0-9]+\s*[章节部分条款]\s*"),
    re.compile(r"^\s*\d+(\.\d+)*\s+"),
]


def _tokenize_ids(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _decode_ids(tokenizer, ids: List[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


class HierarchicalChunker(BaseChunker):
    """基于父子层级的滑动窗口分块器，tokenizer 加载失败时自动降级为字符计数。"""

    def __init__(
        self,
        tokenizer_model_name: str,
        chunk_size: int,
        overlap: int,
        title_patterns: Optional[List[re.Pattern]] = None,
    ) -> None:
        self.tokenizer_model_name = tokenizer_model_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.title_patterns = title_patterns if title_patterns is not None else _DEFAULT_TITLE_PATTERNS

    # ------------------------------------------------------------------
    # BaseChunker 接口实现
    # ------------------------------------------------------------------

    def chunk(self, paragraphs: List[str]) -> ChunkResult:
        parents = self._split_to_parents(paragraphs)
        tokenizer, used_fallback = self._load_tokenizer()
        chunks, parent_child_map = self._slide_window(parents, tokenizer, used_fallback)

        orphans = sum(1 for c in chunks if c["parent_id"] not in parent_child_map)
        avg_len = sum(len(c["text"]) for c in chunks) / max(1, len(chunks))
        stats = {
            "total_parents": len(parents),
            "total_chunks": len(chunks),
            "avg_chunk_length": avg_len,
            "orphans": orphans,
            "used_fallback_tokenizer": used_fallback,
        }
        return ChunkResult(chunks=chunks, parent_child_map=parent_child_map, stats=stats)

    # ------------------------------------------------------------------
    # 内部实现（私有）
    # ------------------------------------------------------------------

    def _is_title_line(self, line: str) -> bool:
        if not line:
            return False
        if any(p.search(line) for p in self.title_patterns):
            return True
        if "：" in line and len(line) <= 40:
            return True
        return False

    def _split_to_parents(self, paragraphs: List[str]) -> List[tuple[str, str]]:
        """按标题行切割为父块列表：[(parent_id, parent_text), ...]"""
        parents: list[tuple[str, str]] = []
        current: list[str] = []
        for p in paragraphs:
            if self._is_title_line(p) and current:
                parent_text = "\n".join(current).strip()
                parents.append((sha256_text(parent_text)[:8], parent_text))
                current = [p]
            else:
                current.append(p)
        if current:
            parent_text = "\n".join(current).strip()
            parents.append((sha256_text(parent_text)[:8], parent_text))
        return parents

    def _load_tokenizer(self):
        try:
            return AutoTokenizer.from_pretrained(self.tokenizer_model_name), False
        except Exception:
            return None, True

    def _slide_window(
        self,
        parents: List[tuple[str, str]],
        tokenizer,
        used_fallback: bool,
    ) -> tuple[list[dict], dict[str, list[str]]]:
        step = max(1, self.chunk_size - self.overlap)
        chunks: list[dict] = []
        parent_child_map: dict[str, list[str]] = {}

        def add_chunk(parent_id: str, chunk_text: str) -> None:
            chunk_id = sha256_text(parent_id + chunk_text)[:12]
            chunks.append({"chunk_id": chunk_id, "parent_id": parent_id, "text": chunk_text})
            parent_child_map.setdefault(parent_id, []).append(chunk_id)

        for parent_id, parent_text in parents:
            if not parent_text.strip():
                continue
            if used_fallback or tokenizer is None:
                tokens = list(parent_text)
                i = 0
                while i < len(tokens):
                    chunk_text = "".join(tokens[i : i + self.chunk_size]).strip()
                    if chunk_text:
                        add_chunk(parent_id, chunk_text)
                    i += step
            else:
                ids = _tokenize_ids(tokenizer, parent_text)
                i = 0
                while i < len(ids):
                    chunk_text = _decode_ids(tokenizer, ids[i : i + self.chunk_size])
                    if chunk_text:
                        add_chunk(parent_id, chunk_text)
                    i += step

        return chunks, parent_child_map



def build_hierarchical_chunks(
    paragraphs: List[str],
    tokenizer_model_name: str,
    chunk_size: int,
    overlap: int,
    title_patterns: Optional[List[re.Pattern]] = None,
) -> ChunkResult:
    chunker = HierarchicalChunker(
        tokenizer_model_name=tokenizer_model_name,
        chunk_size=chunk_size,
        overlap=overlap,
        title_patterns=title_patterns,
    )
    return chunker.chunk(paragraphs)
