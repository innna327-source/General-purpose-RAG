from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import spacy
from spacy.matcher import PhraseMatcher

from config.settings import SETTINGS
from graph.neo4j_store import is_neo4j_enabled, try_load_semantic_graph, try_write_semantic_graph
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


_BATCH_EXTRACT_PROMPT = """\
从下面多个文本块中抽取实体和关系，直接输出 JSON，不要任何解释文字。

输出格式必须严格遵守：
{{
  "items": [
    {{
      "index": 0,
      "entities": [{{"label": "实体名", "type": "类型"}}],
      "relations": [{{"from_label": "实体A", "to_label": "实体B", "relation": "关系描述"}}]
    }}
  ]
}}

要求：
- 每个 items 元素的 index 必须等于输入文本块的 index
- relations 中的实体必须来自同一个文本块的 entities 列表
- 文本中无明确关系时 relations 返回空列表 []
- 不要输出 Markdown 代码块
- 不要遗漏输入中的 index

文本块：
{items}
"""


def _parse_json_object(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(raw)


def _extract_llm(text: str) -> Optional[Dict[str, Any]]:
    """调用 LLM 抽取实体和关系，失败返回 None。"""
    key = SETTINGS.llm_api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=key, base_url=SETTINGS.llm_base_url or None, timeout=90)
        resp = client.chat.completions.create(
            model=SETTINGS.llm_model,
            max_tokens=1024,
            temperature=0.1,
            messages=[
                {"role": "user", "content": _EXTRACT_PROMPT.format(text=text[:2000])}
            ],
        )
        raw = resp.choices[0].message.content.strip()
        return _parse_json_object(raw)
        # 兼容 LLM 用 markdown 代码块包裹 JSON 的情况
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(raw)
    except Exception as exc:
        logger.warning("LLM 实体抽取失败，降级为 spaCy 规则：%s", exc)
        return None


def _extract_llm_batch(items: List[tuple[int, str]]) -> Optional[Dict[int, Dict[str, Any]]]:
    """Batch LLM extraction. Returns {chunk_index: extraction} or None on failure."""
    key = SETTINGS.llm_api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return None

    try:
        from openai import OpenAI

        payload = "\n\n".join(f"[index={idx}]\n{text[:1800]}" for idx, text in items)
        client = OpenAI(api_key=key, base_url=SETTINGS.llm_base_url or None, timeout=90)
        resp = client.chat.completions.create(
            model=SETTINGS.llm_model,
            max_tokens=4096,
            temperature=0.1,
            messages=[
                {"role": "user", "content": _BATCH_EXTRACT_PROMPT.format(items=payload)}
            ],
        )
        raw = resp.choices[0].message.content.strip()
        parsed = _parse_json_object(raw)
        rows = parsed.get("items", [])
        if not isinstance(rows, list):
            return None

        expected = {idx for idx, _ in items}
        output: Dict[int, Dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            idx = row.get("index")
            if idx not in expected:
                continue
            entities = row.get("entities", [])
            relations = row.get("relations", [])
            output[idx] = {
                "entities": entities if isinstance(entities, list) else [],
                "relations": relations if isinstance(relations, list) else [],
            }

        return output if output else None
    except Exception as exc:
        logger.warning("LLM batch extraction failed, will split/fallback: %s", exc)
        return None


def _extract_llm_batch_resilient(items: List[tuple[int, str]]) -> Dict[int, Optional[Dict[str, Any]]]:
    if not items:
        return {}

    batch_result = _extract_llm_batch(items)
    if batch_result is not None:
        return {idx: batch_result.get(idx) for idx, _ in items}

    if len(items) == 1:
        idx, text = items[0]
        return {idx: _extract_llm(text)}

    mid = len(items) // 2
    result = _extract_llm_batch_resilient(items[:mid])
    result.update(_extract_llm_batch_resilient(items[mid:]))
    return result


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


def _extract_llm_for_chunk(index: int, text: str) -> tuple[int, Optional[Dict[str, Any]]]:
    return index, _extract_llm(text)


def _make_batches(chunks: List[dict], batch_size: int) -> List[List[tuple[int, str]]]:
    size = max(1, int(batch_size or 1))
    indexed = [(idx, chunk["text"]) for idx, chunk in enumerate(chunks)]
    return [indexed[i : i + size] for i in range(0, len(indexed), size)]


def _extract_chunks(
    chunks: List[dict],
    max_workers: int = 1,
    progress_every: int = 10,
    batch_size: int = 1,
) -> List[Optional[Dict[str, Any]]]:
    total = len(chunks)
    results: List[Optional[Dict[str, Any]]] = [None] * total
    if total == 0:
        return results

    workers = max(1, int(max_workers or 1))
    if batch_size > 1:
        batches = _make_batches(chunks, batch_size)
        logger.info(
            "semantic graph LLM batch extraction started: chunks=%s batches=%s batch_size=%s workers=%s",
            total,
            len(batches),
            batch_size,
            workers,
        )
        completed = 0
        if workers == 1:
            for batch in batches:
                batch_result = _extract_llm_batch_resilient(batch)
                for idx, result in batch_result.items():
                    results[idx] = result
                completed += len(batch)
                if progress_every and (completed % progress_every == 0 or completed >= total):
                    logger.info("semantic graph LLM extraction progress: %s/%s", min(completed, total), total)
            return results

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_extract_llm_batch_resilient, batch): batch
                for batch in batches
            }
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    batch_result = future.result()
                    for idx, result in batch_result.items():
                        results[idx] = result
                except Exception as exc:
                    logger.warning("LLM batch extraction failed, fallback later: %s", exc)
                completed += len(batch)
                if progress_every and (completed % progress_every == 0 or completed >= total):
                    logger.info("semantic graph LLM extraction progress: %s/%s", min(completed, total), total)
        return results

    if workers == 1:
        for idx, chunk in enumerate(chunks, 1):
            results[idx - 1] = _extract_llm(chunk["text"])
            if progress_every and (idx % progress_every == 0 or idx == total):
                logger.info("semantic graph LLM extraction progress: %s/%s", idx, total)
        return results

    logger.info("semantic graph LLM extraction started: chunks=%s workers=%s", total, workers)
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_extract_llm_for_chunk, idx, chunk["text"]): idx
            for idx, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result_idx, result = future.result()
                results[result_idx] = result
            except Exception as exc:
                logger.warning("LLM 瀹炰綋鎶藉彇澶辫触锛岄檷绾т负 spaCy 瑙勫垯锛?s", exc)
                results[idx] = None
            completed += 1
            if progress_every and (completed % progress_every == 0 or completed == total):
                logger.info("semantic graph LLM extraction progress: %s/%s", completed, total)
    return results


def build_semantic_graph(
    chunks: List[dict],
    file_hash: str,
    graph_dir: Path,
    max_workers: int = 1,
    progress_every: int = 10,
    batch_size: int = 1,
) -> Path:
    ensure_dir(graph_dir)
    out_path = graph_dir / f"{file_hash}_semantic_graph.json"

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
    llm_results = _extract_chunks(
        chunks,
        max_workers=max_workers,
        progress_every=progress_every,
        batch_size=batch_size,
    )
    llm_count = 0
    fallback_count = 0

    def _get_or_create_node(label: str, ent_type: str) -> str:
        nonlocal node_counter
        if label not in label_to_id:
            node_counter += 1
            nid = f"e_{node_counter:03d}"
            label_to_id[label] = nid
            nodes.append({"id": nid, "label": label, "type": ent_type, "attrs": {}})
        return label_to_id[label]

    for c, llm_result in zip(chunks, llm_results):
        cid = c["chunk_id"]

        result = llm_result
        if result is None:
            fallback_count += 1
            result = _extract_spacy(c["text"], nlp, matcher)
        else:
            llm_count += 1

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
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(graph_json, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, out_path)
    neo4j_written = try_write_semantic_graph(file_hash, graph_json, chunks)
    logger.info(
        "semantic graph built: chunks=%s llm=%s fallback=%s nodes=%s edges=%s path=%s neo4j=%s",
        len(chunks),
        llm_count,
        fallback_count,
        len(nodes),
        len(edges),
        out_path,
        neo4j_written,
    )
    return out_path


def load_semantic_graph(path: Path) -> Dict[str, Any]:
    file_hash = path.name.removesuffix("_semantic_graph.json")
    if is_neo4j_enabled():
        graph_data = try_load_semantic_graph(file_hash)
        if graph_data.get("nodes") or graph_data.get("entity_chunks"):
            return graph_data

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}
