import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class RAGQueryRequest(BaseModel):
    question: str


class SimpleContextItem(BaseModel):
    id: str


class SimpleRAGResponse(BaseModel):
    ok: bool
    answer: str
    explanation: str
    related_count: int
    related_items: List[SimpleContextItem]
    raw: Dict[str, Any]


LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "service.log")
DEFAULT_MCP_URL = "http://localhost:8000/v1/completions"  # 你真实的 MCP/RAG HTTP 地址

app = FastAPI(title="RAG Demo API", version="1.0.0")


def log_line(message: str) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {message}\n")


async def call_mcp_rag(question: str) -> Dict[str, Any]:
    url = os.getenv("MCP_RAG_URL", DEFAULT_MCP_URL)
    payload = {"query": question}

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            resp = await client.post(url, json=payload)
        except Exception as e:  # 网络/连接错误
            log_line(f'error calling MCP_RAG_URL="{url}" err="{e}"')
            raise HTTPException(status_code=502, detail="无法连接到后端 RAG 服务，请稍后重试。")

    if resp.status_code != 200:
        log_line(f"mcp_http_error status={resp.status_code} body={resp.text[:500]}")
        raise HTTPException(status_code=502, detail="后端 RAG 服务返回了错误状态码。")

    try:
        data = resp.json()
    except json.JSONDecodeError:
        log_line("mcp_invalid_json")
        raise HTTPException(status_code=502, detail="后端 RAG 服务返回了无效的 JSON。")

    return data


def simplify_mcp_response(raw: Dict[str, Any]) -> SimpleRAGResponse:
    choices: Optional[List[Dict[str, Any]]] = raw.get("choices")
    if not choices:
        return SimpleRAGResponse(
            ok=False,
            answer="抱歉，我现在没有拿到后端的回复。",
            explanation="后端返回的数据里没有找到任何可用答案。",
            related_count=0,
            related_items=[],
            raw=raw,
        )

    first = choices[0]
    text = first.get("text") or ""
    context_ids: List[str] = first.get("context") or []

    cleaned_text = text.replace("\\n", "\n").strip()

    explanation = f"基于 {len(context_ids)} 条相关内容生成的回答。"

    related_items = [SimpleContextItem(id=str(cid)) for cid in context_ids]

    return SimpleRAGResponse(
        ok=True,
        answer=cleaned_text or "抱歉，当前没有生成出可读的回答。",
        explanation=explanation,
        related_count=len(related_items),
        related_items=related_items,
        raw=raw,
    )


@app.post("/api/rag/query", response_model=SimpleRAGResponse)
async def rag_query(req: RAGQueryRequest) -> SimpleRAGResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空。")

    log_line(f'question="{question[:100]}" source="frontend"')

    raw = await call_mcp_rag(question)
    simple = simplify_mcp_response(raw)

    log_line(
        f'answer_len={len(simple.answer)} related_count={simple.related_count} '
        f'ok={simple.ok}'
    )

    return simple


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}

