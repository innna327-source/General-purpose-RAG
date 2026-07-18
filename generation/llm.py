from __future__ import annotations

import os
import json
import re
import unicodedata
from typing import Optional

from config.settings import SETTINGS
from generation.constants import NO_ANSWER_PHRASES, NO_ANSWER_RESPONSE
from retrieval.evidence_answerability import check_direct_answer_evidence
from retrieval.retrieval_gate import (
    check_answer_claim_support,
    check_retrieval_confidence,
    check_retrieval_sufficiency,
)


_SYSTEM_PROMPT = """\
你是一个严谨的文档问答助手。请严格依据下方【参考资料】回答用户问题。
规则：
1. 只使用参考资料中明确出现的信息，不得推断或补充资料外的内容。
2. 如果参考资料不能直接支持答案，必须只回复“根据现有资料无法回答该问题”。
3. 每个结论都必须能在参考资料中找到对应依据；不要给常识性、经验性或泛泛而谈的回答。
4. 仅“话题相关”不等于“可以回答”；必须有直接证据才能回答。
5. 只输出 JSON，不要输出 Markdown。格式示例：
{"answerable": true, "answer": "答案文本"}
或
{"answerable": false, "answer": "根据现有资料无法回答该问题"}
6. 当 answerable 为 false 时，answer 必须是“根据现有资料无法回答该问题”。
"""

_UNCERTAIN_EVIDENCE_WARNING = (
    "提示：召回内容与问题主题相关，但没有完全覆盖问题中的关键维度；"
    "以下回答仅基于现有片段，可能不完整。"
)

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
    "一下",
    "请问",
    "为什么",
    "多少",
}


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    return re.sub(r"\s+", "", text)


def _content_terms(text: str) -> list[str]:
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


def _term_coverage(terms: list[str], text: str) -> float:
    if not terms:
        return 1.0
    norm_text = _normalize(text)
    hits = sum(1 for term in terms if _normalize(term) in norm_text)
    return hits / len(terms)


def is_no_answer_response(response: str) -> bool:
    return any(phrase in (response or "") for phrase in NO_ANSWER_PHRASES)


def is_context_answerable(query: str, context_text: str) -> tuple[bool, dict]:
    query_terms = _content_terms(query)
    coverage = _term_coverage(query_terms, context_text)
    min_coverage = float(getattr(SETTINGS, "answerability_min_query_coverage", 0.30))
    answerable = bool(context_text.strip()) and coverage >= min_coverage
    return answerable, {
        "query_terms": query_terms,
        "query_coverage": round(coverage, 4),
        "min_query_coverage": min_coverage,
    }


def is_answer_supported(answer: str, context_text: str) -> tuple[bool, dict]:
    if is_no_answer_response(answer):
        return True, {"answer_coverage": 1.0, "checked_terms": []}

    answer_terms = _content_terms(answer)
    # Keep the verifier focused on claims. Terms already present in the question
    # are less useful for detecting unsupported additions, so the context check
    # is intentionally run against the answer itself.
    coverage = _term_coverage(answer_terms, context_text)
    min_coverage = float(getattr(SETTINGS, "answerability_min_answer_coverage", 0.50))
    return coverage >= min_coverage, {
        "answer_coverage": round(coverage, 4),
        "min_answer_coverage": min_coverage,
        "checked_terms": answer_terms,
    }


def _parse_answer_payload(raw: str) -> tuple[Optional[bool], str]:
    text = (raw or "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None, text

    try:
        payload = json.loads(match.group())
    except Exception:
        return None, text

    answerable = payload.get("answerable")
    if not isinstance(answerable, bool):
        answerable = None
    answer = payload.get("answer", "")
    if not isinstance(answer, str):
        answer = str(answer)
    return answerable, answer.strip()



def _clean_answer_text(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw or "").strip()
    if not text:
        return text

    answerable, parsed_answer = _parse_answer_payload(text)
    if answerable is False:
        return NO_ANSWER_RESPONSE
    if parsed_answer and parsed_answer != text:
        text = parsed_answer
    else:
        lower_text = text.lower()
        answer_idx = lower_text.find("\nanswer:")
        answer_marker_len = len("\nanswer:")
        if answer_idx < 0 and lower_text.startswith("answer:"):
            answer_idx = 0
            answer_marker_len = len("answer:")
        if answer_idx >= 0:
            text = text[answer_idx + answer_marker_len :].strip()

    text = re.sub(r'^\s*["\'{]+', "", text)
    text = re.sub(r'["\'}]+\s*$', "", text)
    text = re.sub(r'^\s*answerable\s*:.*?\n+', "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'\n?\s*["\']?\d+\s*([^"\n]{2,20})["\']?\s*:', r"\n\n\1:", text)
    text = re.sub(r'\n{3,}', "\n\n", text)
    return text.strip()


def _with_uncertain_evidence_warning(answer: str) -> str:
    if not answer or is_no_answer_response(answer):
        return answer
    if answer.startswith(_UNCERTAIN_EVIDENCE_WARNING):
        return answer
    return f"{_UNCERTAIN_EVIDENCE_WARNING}\n\n{answer}"


def generate_answer(
    query: str,
    context_text: str,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    enforce_answerability: Optional[bool] = None,
    system_prompt: Optional[str] = None,
    retrieval_results: Optional[list[dict]] = None,
    high_risk: bool = False,
    enable_llm_judge: Optional[bool] = None,
) -> str:
    """
    调用 OpenAI 兼容接口，根据检索到的 context_text 生成答案。
    启用 gate 时，生成前检查召回证据，生成后检查答案是否被上下文支持。
    """
    use_gate = (
        SETTINGS.enable_answerability_gate
        if enforce_answerability is None
        else enforce_answerability
    )
    uncertain_evidence = False

    if use_gate:
        if retrieval_results is not None:
            sufficient, _ = check_retrieval_sufficiency(
                query=query,
                retrieval_results=retrieval_results,
                context_text=context_text,
            )
            confident, _ = check_retrieval_confidence(
                query=query,
                retrieval_results=retrieval_results,
                context_text=context_text,
            )
            if not sufficient or not confident:
                return NO_ANSWER_RESPONSE
            if getattr(SETTINGS, "enable_direct_evidence_answerability_gate", True):
                direct_answerable, _ = check_direct_answer_evidence(
                    query=query,
                    retrieval_results=retrieval_results,
                    context_text=context_text,
                )
                if not direct_answerable:
                    if high_risk or getattr(SETTINGS, "answerability_uncertain_normal_policy", "warn") == "reject":
                        return NO_ANSWER_RESPONSE
                    uncertain_evidence = True
        else:
            answerable, _ = is_context_answerable(query, context_text)
            if not answerable:
                return NO_ANSWER_RESPONSE

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return context_text
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("未安装 openai 包，请运行：pip install openai")

    user_message = f"【参考资料】\n{context_text}\n\n【问题】\n{query}"

    client = OpenAI(api_key=key, base_url=base_url or None)
    response = client.chat.completions.create(
        model=model,
        max_tokens=int(getattr(SETTINGS, "generation_max_tokens", 512)),
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt or _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    answer = _clean_answer_text(response.choices[0].message.content)

    if use_gate:
        supported, _ = is_answer_supported(answer, context_text)
        if not supported:
            return NO_ANSWER_RESPONSE
        strict_claim_check = high_risk or getattr(SETTINGS, "domain_profile", "general") == "finance"
        if strict_claim_check:
            claims_supported, _ = check_answer_claim_support(answer, context_text)
            if not claims_supported:
                return NO_ANSWER_RESPONSE
        if high_risk and (enable_llm_judge or getattr(SETTINGS, "enable_high_risk_llm_judge", False)):
            # Keep this hook explicit; wire a separate judge model only for high-risk report paths.
            pass

    if uncertain_evidence:
        answer = _with_uncertain_evidence_warning(answer)

    return answer
