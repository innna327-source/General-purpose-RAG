from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import spacy
from spacy.matcher import PhraseMatcher

from config.settings import SETTINGS
from utils.paths import ensure_dir

logger = logging.getLogger(__name__)

DEFAULT_TECH_TERMS = [
    "RAG",
    "GraphRAG",
    "BM25",
    "FAISS",
    "OCR",
    "Tesseract",
    "spaCy",
    "PyMuPDF",
    "SentenceTransformer",
    "all-MiniLM-L6-v2",
    "高血压",
    "糖尿病",
]

# 要求 LLM 只输出 JSON，不夹杂解释文字
_EXTRACT_PROMPT = """\
从以下文本中抽取实体和关系，直接输出 JSON，不要任何解释文字。

输出格式（严格遵守，只输出这一个 JSON 对象）：
{{"entities": [{{"label": "实体名", "type": "类型"}}], "relations": [{{"from_label": "实体A", "to_label": "实体B", "relation": "关系描述"}}]}}

说明：
- 实体类型自行判断，例如 PERSON（人物）、ORG（组织/机构）、TASK（任务/项目）、EVENT（事件）、CONCEPT（概念/技术）等
- relations 中的实体必须来自 entities 列表
- 文本中无明确关系时 relations 返回空列表 []

文本：
{text}"""


def _load_nlp():
    try:
        return spacy.load("zh_core_web_sm")
    except Exception:
        return spacy.blank("zh")


def _extract_llm(text: str) -> Optional[Dict[str, Any]]:
    """调用 LLM 抽取实体和关系，失败返回 None。"""
    key = SETTINGS.llm_api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=key, base_url=SETTINGS.llm_base_url or None)
        resp = client.chat.completions.create(
            model=SETTINGS.llm_model,
            max_tokens=1024,
            temperature=0.1,
            messages=[
                {"role": "user", "content": _EXTRACT_PROMPT.format(text=text[:2000])}
            ],
        )
        raw = resp.choices[0].message.content.strip()
        # 兼容 LLM 用 markdown 代码块包裹 JSON 的情况
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(raw)
    except Exception as exc:
        logger.warning("LLM 实体抽取失败，降级为 spaCy 规则：%s", exc)
        return None


def _extract_spacy(text: str, nlp, matcher) -> Dict[str, Any]:
    """spaCy 规则抽取（LLM 降级）。只产出实体，不产出关系。"""
    doc = nlp(text)
    ents: set[str] = set()

    for e in getattr(doc, "ents", []):
        if e.text.strip():
            ents.add(e.text.strip())

    for _, start, end in matcher(doc):
        span = doc[start:end]
        if span.text.strip():
            ents.add(span.text.strip())

    # 最小保障：英文/数字+字母串
    for token in doc:
        t = token.text.strip()
        if len(t) >= 2 and any(ch.isalpha() for ch in t):
            ents.add(t)

    return {
        "entities": [{"label": e, "type": "CONCEPT"} for e in ents],
        "relations": [],
    }


def build_semantic_graph(chunks: List[dict], file_hash: str, graph_dir: Path) -> Path:
    ensure_dir(graph_dir)
    out_path = graph_dir / f"{file_hash}_semantic_graph.json"
    if out_path.exists():
        out_path.unlink(missing_ok=True)

    nlp = _load_nlp()
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(t) for t in DEFAULT_TECH_TERMS if t]
    if patterns:
        matcher.add("TECH_TERMS", patterns)

    # label → node_id（跨 chunk 同名实体合并）
    label_to_id: Dict[str, str] = {}
    nodes: List[dict] = []
    entity_chunks: Dict[str, set[str]] = {}
    # (from_id, to_id, relation) → 出现次数（即 weight）
    edge_counter: Dict[tuple[str, str, str], int] = {}
    node_counter = 0

    def _get_or_create_node(label: str, ent_type: str) -> str:
        nonlocal node_counter
        if label not in label_to_id:
            node_counter += 1
            nid = f"e_{node_counter:03d}"
            label_to_id[label] = nid
            nodes.append({"id": nid, "label": label, "type": ent_type, "attrs": {}})
        return label_to_id[label]

    for c in chunks:
        cid = c["chunk_id"]
        text = c["text"]

        result = _extract_llm(text)
        if result is None:
            result = _extract_spacy(text, nlp, matcher)

        # 先处理所有实体，确保关系引用的标签都已注册
        for ent in result.get("entities", []):
            label = ent.get("label", "").strip()
            ent_type = ent.get("type", "CONCEPT").strip() or "CONCEPT"
            if not label:
                continue
            nid = _get_or_create_node(label, ent_type)
            entity_chunks.setdefault(nid, set()).add(cid)

        # 再处理关系：只用 LLM 明确抽取的关系建边
        for rel in result.get("relations", []):
            from_label = rel.get("from_label", "").strip()
            to_label = rel.get("to_label", "").strip()
            relation = rel.get("relation", "").strip()
            if not from_label or not to_label or not relation:
                continue
            # 关系中的标签必须来自本次已注册的实体
            if from_label not in label_to_id or to_label not in label_to_id:
                continue
            key = (label_to_id[from_label], label_to_id[to_label], relation)
            edge_counter[key] = edge_counter.get(key, 0) + 1

    edges = [
        {"from": f, "to": t, "relation": r, "weight": w}
        for (f, t, r), w in sorted(edge_counter.items(), key=lambda x: x[1], reverse=True)
    ]

    graph_json: Dict[str, Any] = {
        "nodes": nodes,
        "edges": edges,
        "entity_chunks": {k: sorted(list(v)) for k, v in entity_chunks.items()},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(graph_json, f, ensure_ascii=False, indent=2)
    return out_path


def load_semantic_graph(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}
