from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
from transformers import AutoTokenizer

from chunk.base_chunker import BaseChunker, ChunkResult
from utils.hash_utils import sha256_text


# ---------------------------------------------------------------------------
# 默认标题正则：通用中文文档结构
# 金融报告、法律文书等可在构造时传入自己的 title_patterns 覆盖
# ---------------------------------------------------------------------------
_DEFAULT_TITLE_PATTERNS: List[re.Pattern] = [
    re.compile(r"^\s*#"),
    re.compile(r"^\s*第\s*[一二三四五六七八九十0-9]+\s*[章节部分条款]\s*"),
    re.compile(r"^\s*\d+(\.\d+)*\s+"),
]

# 正则切分产出的父块数少于此值时，自动切换到语义聚类模式
# 简历、论文等依靠字号而非文字标记层级的文档，正则通常只切出 2-5 个块，设为 8 可确保降级
_MIN_PARENTS_REGEX = 2


def _tokenize_ids(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _decode_ids(tokenizer, ids: List[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class HierarchicalChunker(BaseChunker):
    """
    基于父子层级的滑动窗口分块器。

    父块切分策略（按优先级）：
      1. 标题正则：识别显式章节标题，适合有结构标题的文档。
      2. 语义边界检测（自动降级）：当正则切出的父块 < _MIN_PARENTS_REGEX 时启用。
         对所有段落做 embedding，计算相邻段落余弦相似度，
         在相似度"低谷"处划定主题边界，生成语义化父块。

    title_patterns：识别"父块边界"的正则列表，默认覆盖通用中文章节结构。
    tokenizer_model_name：用于精确 token 计数，加载失败自动 fallback 到字符计数。
    embedding_model_name：语义切分时使用的 SentenceTransformer 模型，
                          默认与 tokenizer 同名（bge 系列兼容）。
    chunk_size / overlap：以 token 数计的窗口大小和重叠量。
    semantic_threshold：相邻段落相似度低于此值时视为主题边界（0~1）。
                        None = 自适应（均值 - 0.5×标准差）。
    """

    def __init__(
        self,
        tokenizer_model_name: str,
        chunk_size: int,
        overlap: int,
        title_patterns: Optional[List[re.Pattern]] = None,
        embedding_model_name: Optional[str] = None,
        semantic_threshold: Optional[float] = None,
    ) -> None:
        self.tokenizer_model_name = tokenizer_model_name
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.title_patterns = title_patterns if title_patterns is not None else _DEFAULT_TITLE_PATTERNS
        self.embedding_model_name = embedding_model_name or tokenizer_model_name
        self.semantic_threshold = semantic_threshold
        self._embedder = None

    # ------------------------------------------------------------------
    # BaseChunker 接口实现
    # ------------------------------------------------------------------

    def chunk(self, paragraphs: List[str]) -> ChunkResult:
        # 先尝试正则切父块
        parents = self._split_to_parents_regex(paragraphs)
        split_method = "regex"

        # 正则切出的父块太少，降级为语义边界检测
        if len(parents) < _MIN_PARENTS_REGEX:
            parents = self._split_to_parents_semantic(paragraphs)
            split_method = "semantic"

        tokenizer, used_fallback = self._load_tokenizer()
        chunks, parent_child_map = self._slide_window(parents, tokenizer, used_fallback)

        orphans = sum(1 for c in chunks if c["parent_id"] not in parent_child_map)
        lengths = [len(c["text"]) for c in chunks]
        avg_len = sum(lengths) / max(1, len(lengths))
        sorted_len = sorted(lengths)
        n = len(sorted_len)

        def _pct(p: float) -> float:
            if n == 0:
                return 0.0
            idx = int(n * p)
            return float(sorted_len[min(idx, n - 1)])

        cpp = [len(v) for v in parent_child_map.values()]
        avg_cpp = sum(cpp) / max(1, len(cpp))

        stats = {
            "total_parents": len(parents),
            "total_chunks": len(chunks),
            "avg_chunk_length": round(avg_len, 1),
            "min_chunk_length": sorted_len[0] if sorted_len else 0,
            "max_chunk_length": sorted_len[-1] if sorted_len else 0,
            "p25_chunk_length": _pct(0.25),
            "p50_chunk_length": _pct(0.50),
            "p75_chunk_length": _pct(0.75),
            "avg_chunks_per_parent": round(avg_cpp, 1),
            "orphans": orphans,
            "used_fallback_tokenizer": used_fallback,
            "parent_split_method": split_method,
            "chunk_size_setting": self.chunk_size,
            "overlap_setting": self.overlap,
        }
        return ChunkResult(chunks=chunks, parent_child_map=parent_child_map, stats=stats)

    # ------------------------------------------------------------------
    # 父块切分：正则模式
    # ------------------------------------------------------------------

    def _is_title_line(self, line: str) -> bool:
        if not line:
            return False
        if any(p.search(line) for p in self.title_patterns):
            return True
        if "：" in line and len(line) <= 40:
            return True
        return False

    def _split_to_parents_regex(self, paragraphs: List[str]) -> List[tuple[str, str]]:
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

    # ------------------------------------------------------------------
    # 父块切分：语义边界检测模式
    # ------------------------------------------------------------------

    def _split_to_parents_semantic(self, paragraphs: List[str]) -> List[tuple[str, str]]:
        """
        对段落序列做 embedding，计算相邻段落余弦相似度，
        在相似度低谷处划定主题边界，将连续段落聚合为语义父块。

        原理：同一主题内相邻段落语义接近（高相似度），
              主题切换处相似度骤降，形成低谷，即为边界。
        """
        if len(paragraphs) <= 1:
            text = paragraphs[0] if paragraphs else ""
            return [(sha256_text(text)[:8], text)]

        try:
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.embedding_model_name, device="cpu")
            vecs = self._embedder.encode(
                paragraphs,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception:
            # embedding 失败：均分分组，避免所有子块挂在同一超大父块下
            n_groups = 2 if len(paragraphs) <= 6 else 3
            size = (len(paragraphs) + n_groups - 1) // n_groups  # ceiling division
            result = []
            for i in range(0, len(paragraphs), size):
                parent_text = "\n".join(paragraphs[i : i + size]).strip()
                if parent_text:
                    result.append((sha256_text(parent_text)[:8], parent_text))
            return result if result else [(sha256_text("")[:8], "\n".join(paragraphs))]

        # 计算相邻段落相似度
        sims = np.array([
            _cosine_sim(vecs[i], vecs[i + 1])
            for i in range(len(vecs) - 1)
        ])

        # 边界阈值：自适应（均值 - 0.5×标准差）或用户指定
        if self.semantic_threshold is not None:
            threshold = self.semantic_threshold
        else:
            threshold = float(sims.mean() - 0.5 * sims.std())

        # 找边界索引（相似度低谷处，i 表示第 i 和 i+1 段之间）
        boundaries: set[int] = set()
        for i, s in enumerate(sims):
            if s < threshold:
                boundaries.add(i + 1)  # i+1 是新主题块的起始段落

        # 按边界聚合段落为父块
        parents: list[tuple[str, str]] = []
        start = 0
        sorted_boundaries = sorted(boundaries)
        for boundary in sorted_boundaries:
            group = paragraphs[start:boundary]
            if group:
                parent_text = "\n".join(group).strip()
                parents.append((sha256_text(parent_text)[:8], parent_text))
            start = boundary
        # 最后一组
        if start < len(paragraphs):
            parent_text = "\n".join(paragraphs[start:]).strip()
            parents.append((sha256_text(parent_text)[:8], parent_text))

        return parents if parents else [(sha256_text("")[:8], "\n".join(paragraphs))]

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    def _load_tokenizer(self):
        try:
            return AutoTokenizer.from_pretrained(self.tokenizer_model_name), False
        except Exception:
            return None, True

    # ------------------------------------------------------------------
    # 子块：滑动窗口
    # ------------------------------------------------------------------

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
                # 使用 offset_mapping 直接切原始文本，避免 decode() 在中文字符间插入空格
                try:
                    enc = tokenizer(
                        parent_text,
                        add_special_tokens=False,
                        return_offsets_mapping=True,
                        truncation=False,
                    )
                    token_ids = enc["input_ids"]
                    offsets = enc["offset_mapping"]
                    i = 0
                    while i < len(token_ids):
                        end = min(i + self.chunk_size, len(token_ids))
                        char_start = offsets[i][0]
                        char_end = offsets[end - 1][1]
                        chunk_text = parent_text[char_start:char_end].strip()
                        if chunk_text:
                            add_chunk(parent_id, chunk_text)
                        i += step
                except Exception:
                    # fast tokenizer 不可用时降级为字符切分
                    tokens = list(parent_text)
                    i = 0
                    while i < len(tokens):
                        chunk_text = "".join(tokens[i : i + self.chunk_size]).strip()
                        if chunk_text:
                            add_chunk(parent_id, chunk_text)
                        i += step

        return chunks, parent_child_map


# ---------------------------------------------------------------------------
# 向后兼容的函数式入口（保留给 main.py 的旧调用，内部委托给 HierarchicalChunker）
# ---------------------------------------------------------------------------

def build_hierarchical_chunks(
    paragraphs: List[str],
    tokenizer_model_name: str,
    chunk_size: int,
    overlap: int,
    title_patterns: Optional[List[re.Pattern]] = None,
    embedding_model_name: Optional[str] = None,
    semantic_threshold: Optional[float] = None,
) -> ChunkResult:
    chunker = HierarchicalChunker(
        tokenizer_model_name=tokenizer_model_name,
        chunk_size=chunk_size,
        overlap=overlap,
        title_patterns=title_patterns,
        embedding_model_name=embedding_model_name,
        semantic_threshold=semantic_threshold,
    )
    return chunker.chunk(paragraphs)
