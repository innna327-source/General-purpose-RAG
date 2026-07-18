from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any

from config.settings import SETTINGS


_STOPWORDS = {
    "什么",
    "怎么",
    "如何",
    "哪个",
    "哪些",
    "是否",
    "是不是",
    "可以",
    "能够",
    "需要",
    "以及",
    "还是",
    "这个",
    "那个",
    "一个",
    "请问",
    "为什么",
    "多少",
}

_FINANCE_METRICS = (
    "营业收入",
    "营收",
    "收入",
    "净利润",
    "归母净利润",
    "毛利率",
    "现金流",
    "经营活动现金流",
    "资产负债率",
    "负债率",
    "研发投入",
    "研发费用",
    "风险",
    "合规",
    "减值",
)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    return re.sub(r"\s+", "", text)


def content_terms(text: str) -> list[str]:
    try:
        import jieba

        raw_terms = jieba.cut(text or "")
    except Exception:
        raw_terms = re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9][a-zA-Z0-9_.+-]*", text or "")

    terms: list[str] = []
    seen: set[str] = set()
    for term in raw_terms:
        t = unicodedata.normalize("NFKC", str(term)).strip().lower()
        if not t or t in _STOPWORDS:
            continue
        if len(t) < 2 and not re.fullmatch(r"[a-z0-9]+", t):
            continue
        if re.fullmatch(r"[\W_]+", t):
            continue
        if t not in seen:
            seen.add(t)
            terms.append(t)
    return terms


def term_coverage(terms: list[str], text: str) -> float:
    if not terms:
        return 1.0
    norm_text = normalize_text(text)
    hits = sum(1 for term in terms if normalize_text(term) in norm_text)
    return hits / len(terms)


def is_direct_evidence(row: dict[str, Any]) -> bool:
    evidence_type = str(row.get("evidence_type", "")).lower()
    if evidence_type in {"direct", "direct_evidence"}:
        return True
    return float(row.get("bm25_score", 0.0) or 0.0) > 0 or float(row.get("vector_score", 0.0) or 0.0) > 0


def _result_text(results: list[dict[str, Any]] | None, fallback: str) -> str:
    if not results:
        return fallback or ""
    return "\n\n".join(str(row.get("text", "")) for row in results)


def check_retrieval_sufficiency(
    query: str,
    retrieval_results: list[dict[str, Any]] | None = None,
    context_text: str = "",
) -> tuple[bool, dict[str, Any]]:
    evidence_text = _result_text(retrieval_results, context_text)
    query_terms = content_terms(query)
    coverage = term_coverage(query_terms, evidence_text)
    min_coverage = float(getattr(SETTINGS, "answerability_min_query_coverage", 0.30))
    min_chars = int(getattr(SETTINGS, "answerability_min_context_chars", 80))

    reasons: list[str] = []
    topk_count = len(retrieval_results) if retrieval_results is not None else None
    direct_count = None
    related_count = None

    if retrieval_results is not None:
        direct_count = sum(1 for row in retrieval_results if is_direct_evidence(row))
        related_count = sum(
            1
            for row in retrieval_results
            if str(row.get("evidence_type", "")).lower() in {"related_context", "graph_only"}
        )
        if not retrieval_results:
            reasons.append("empty_topk")
        if direct_count == 0:
            reasons.append("no_direct_evidence")

    if len(evidence_text.strip()) < min_chars:
        reasons.append("context_too_short")
    if coverage < min_coverage:
        reasons.append("low_query_coverage")

    return not reasons, {
        "gate": "retrieval_sufficiency",
        "passed": not reasons,
        "reasons": reasons,
        "topk_count": topk_count,
        "context_chars": len(evidence_text.strip()),
        "min_context_chars": min_chars,
        "query_terms": query_terms,
        "query_coverage": round(coverage, 4),
        "min_query_coverage": min_coverage,
        "direct_evidence_count": direct_count,
        "related_context_count": related_count,
    }


def _score_gap(top: dict[str, Any], second: dict[str, Any] | None) -> float:
    if second is None:
        return 1.0
    if top.get("rerank_used") and second.get("rerank_used"):
        return abs(float(top.get("rerank_score", 0.0) or 0.0) - float(second.get("rerank_score", 0.0) or 0.0))
    return abs(float(top.get("final_score", 0.0) or 0.0) - float(second.get("final_score", 0.0) or 0.0))


def _rerank_relevance(score: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, score))))


def check_retrieval_confidence(
    query: str,
    retrieval_results: list[dict[str, Any]] | None = None,
    context_text: str = "",
) -> tuple[bool, dict[str, Any]]:
    if not retrieval_results:
        return False, {
            "gate": "retrieval_confidence",
            "passed": False,
            "retrieval_confidence": 0.0,
            "reasons": ["empty_topk"],
        }

    rows = list(retrieval_results)
    top1 = rows[0]
    top2 = rows[1] if len(rows) > 1 else None
    direct_count = sum(1 for row in rows if is_direct_evidence(row))
    direct_ratio = direct_count / len(rows)
    related_ratio = sum(
        1
        for row in rows
        if str(row.get("evidence_type", "")).lower() in {"related_context", "graph_only"}
    ) / len(rows)

    if top1.get("rerank_used"):
        top1_score = float(top1.get("rerank_score", 0.0) or 0.0)
        top1_component = _rerank_relevance(top1_score)
        min_top1_score = float(getattr(SETTINGS, "answerability_min_top1_rerank_score", -3.0))
    else:
        top1_score = float(top1.get("final_score", 0.0) or 0.0)
        top1_component = max(0.0, min(1.0, top1_score))
        min_top1_score = float(getattr(SETTINGS, "answerability_min_top1_final_score", 0.05))

    gap = _score_gap(top1, top2)
    gap_component = max(0.0, min(1.0, gap / 0.15))

    sources = [
        str(row.get("source_file") or row.get("file_hash") or "single-document")
        for row in rows
    ]
    parents = [str(row.get("parent_id") or "") for row in rows if row.get("parent_id")]
    top_source_ratio = Counter(sources).most_common(1)[0][1] / len(rows) if sources else 0.0
    top_parent_ratio = Counter(parents).most_common(1)[0][1] / len(parents) if parents else 0.0
    source_component = 1.0 if top_source_ratio >= 0.4 else 0.5
    parent_crowding_penalty = 0.15 if len(rows) >= 3 and top_parent_ratio >= 0.8 else 0.0

    confidence = (
        0.35 * direct_ratio
        + 0.25 * top1_component
        + 0.15 * gap_component
        + 0.15 * source_component
        + 0.10 * (1.0 - related_ratio)
        - parent_crowding_penalty
    )
    confidence = max(0.0, min(1.0, confidence))
    min_confidence = float(getattr(SETTINGS, "answerability_min_retrieval_confidence", 0.35))

    reasons: list[str] = []
    if top1_score < min_top1_score:
        reasons.append("low_top1_score")
    if direct_ratio <= 0:
        reasons.append("no_direct_evidence")
    if related_ratio >= float(getattr(SETTINGS, "answerability_max_related_context_ratio", 0.80)):
        reasons.append("too_much_related_context")
    if confidence < min_confidence:
        reasons.append("low_retrieval_confidence")

    return not reasons, {
        "gate": "retrieval_confidence",
        "passed": not reasons,
        "reasons": reasons,
        "retrieval_confidence": round(confidence, 4),
        "min_retrieval_confidence": min_confidence,
        "top1_score": round(top1_score, 4),
        "min_top1_score": min_top1_score,
        "top1_top2_gap": round(gap, 4),
        "direct_evidence_ratio": round(direct_ratio, 4),
        "related_context_ratio": round(related_ratio, 4),
        "top_source_ratio": round(top_source_ratio, 4),
        "top_parent_ratio": round(top_parent_ratio, 4),
    }


def extract_finance_claim_terms(text: str) -> dict[str, list[str]]:
    years = sorted(set(re.findall(r"(?:19|20)\d{2}\s*年?", text or "")))
    amounts = sorted(
        set(
            re.findall(
                r"\d+(?:\.\d+)?\s*(?:亿元|万元|元|亿|万|%|％|pct|百分点|倍)",
                text or "",
                flags=re.IGNORECASE,
            )
        )
    )
    metrics = [metric for metric in _FINANCE_METRICS if metric in (text or "")]
    companies = sorted(
        set(
            re.findall(
                r"[\u4e00-\u9fffA-Za-z0-9]{2,30}(?:股份有限公司|有限公司|集团|公司|银行|证券|科技)",
                text or "",
            )
        )
    )
    risks = sorted(set(re.findall(r"[\u4e00-\u9fff]{2,12}风险", text or "")))
    return {
        "companies": companies,
        "metrics": metrics,
        "years": years,
        "amounts": amounts,
        "risks": risks,
    }


def check_answer_claim_support(answer: str, context_text: str) -> tuple[bool, dict[str, Any]]:
    claims = extract_finance_claim_terms(answer)
    unsupported: dict[str, list[str]] = {}
    norm_context = normalize_text(context_text)

    for claim_type, values in claims.items():
        missing = [value for value in values if normalize_text(value) not in norm_context]
        if missing:
            unsupported[claim_type] = missing

    total_claims = sum(len(values) for values in claims.values())
    missing_claims = sum(len(values) for values in unsupported.values())
    passed = missing_claims == 0
    return passed, {
        "gate": "answer_claim_support",
        "passed": passed,
        "total_claims": total_claims,
        "missing_claims": missing_claims,
        "claims": claims,
        "unsupported_claims": unsupported,
    }
