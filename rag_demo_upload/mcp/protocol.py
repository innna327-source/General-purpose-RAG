from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple


ERROR_INVALID_REQUEST = 40001
ERROR_RATE_LIMIT = 40002
ERROR_INTERNAL = 50001


@dataclass
class MCPRequest:
    query: str
    max_tokens: int = 600  # 本项目中定义为“最大字符数”
    top_k: int = 5
    model: str | None = None


def parse_request(payload: Dict[str, Any]) -> Tuple[MCPRequest | None, Dict[str, Any] | None]:
    if not isinstance(payload, dict):
        return None, make_error(ERROR_INVALID_REQUEST, "Invalid JSON object")
    query = payload.get("query")
    if not query or not isinstance(query, str):
        return None, make_error(ERROR_INVALID_REQUEST, "Field `query` is required")

    max_tokens = payload.get("max_tokens", 600)
    try:
        max_tokens = int(max_tokens)
    except Exception:
        return None, make_error(ERROR_INVALID_REQUEST, "Field `max_tokens` must be int")
    if not (1 <= max_tokens <= 1000):
        return None, make_error(ERROR_INVALID_REQUEST, "Field `max_tokens` must be in [1, 1000]")

    top_k = payload.get("top_k", 5)
    try:
        top_k = int(top_k)
    except Exception:
        top_k = 5
    if top_k <= 0:
        top_k = 5

    model = payload.get("model")
    if model is not None and not isinstance(model, str):
        model = None

    return MCPRequest(query=query, max_tokens=max_tokens, top_k=top_k, model=model), None


def build_completion_response(service_name: str, text: str, context_ids: list[str], request_model: str | None = None) -> Dict[str, Any]:
    return {
        "id": f"mcp-rag-{int(time.time() * 1000)}",
        "model": request_model or service_name,
        "object": "text_completion",
        "created": int(time.time()),
        "choices": [
            {
                "index": 0,
                "text": text,
                "context": context_ids,
            }
        ],
    }


def apply_max_tokens_concat(chunks: list[dict], max_tokens: int, sep: str = "\n\n---\n\n") -> tuple[str, list[str]]:
    # 按排序依次拼接为 text；若超过 max_tokens，则从末尾截断
    context_ids: list[str] = [c["chunk_id"] for c in chunks]
    text = sep.join([c["text"] for c in chunks])
    if len(text) > max_tokens:
        text = text[:max_tokens]
    return text, context_ids


def make_error(code: int, message: str) -> Dict[str, Any]:
    return {"error": {"code": code, "message": message}}

