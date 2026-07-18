from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from config.settings import SETTINGS
from generation.llm import generate_answer
from mcp.protocol import (
    ERROR_INTERNAL,
    MCPRequest,
    apply_max_tokens_concat,
    build_completion_response,
    make_error,
    parse_request,
)
from retrieval.hybrid import HybridRetriever
from utils import logger


@dataclass
class MCPHandler:
    retriever: HybridRetriever
    service_name: str
    keep_context_n: int = 3

    def __post_init__(self) -> None:
        self._recent_contexts = deque(maxlen=self.keep_context_n)

    def handle_completions(self, payload: Dict[str, Any]) -> tuple[Dict[str, Any], int]:
        req, err = parse_request(payload)
        if err is not None:
            return err, 400

        assert req is not None
        t0 = time.time()
        try:
            results, _ = self.retriever.hybrid_retrieve(req.query, top_k=req.top_k, return_debug_info=False)
            context_text, context_ids = apply_max_tokens_concat(results, max_tokens=req.max_tokens)
            self._recent_contexts.append(context_ids)
            answer = generate_answer(
                query=req.query,
                context_text=context_text,
                model=SETTINGS.generation_llm_model,
                api_key=SETTINGS.llm_api_key or None,
                base_url=SETTINGS.llm_base_url or None,
                retrieval_results=results,
            )
            resp = build_completion_response(self.service_name, text=answer, context_ids=context_ids, request_model=req.model)
            latency_ms = (time.time() - t0) * 1000
            logger.log_service_summary(req.query, top_k=req.top_k, latency_ms=latency_ms)
            return resp, 200
        except Exception as e:
            latency_ms = (time.time() - t0) * 1000
            logger.error(f"MCP handler internal error: {e}")
            logger.log_service_summary(req.query, top_k=req.top_k, latency_ms=latency_ms)
            return make_error(ERROR_INTERNAL, "Internal processing error"), 500

