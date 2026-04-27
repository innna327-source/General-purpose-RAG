import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class RAGQueryRequest(BaseModel):
    """前端发来的提问请求"""

    question: str


class SimpleContextItem(BaseModel):
    """简化后的上下文信息（给前端展示用）"""

    id: str


class SimpleRAGResponse(BaseModel):
    """后端返回给前端的友好结构，避免一堆难懂缩写"""

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
    """简单写日志到 logs/service.log，方便用户排查"""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {message}\n")


async def call_mcp_rag(question: str) -> Dict[str, Any]:
    """
    调用已有的 MCP/RAG HTTP 服务。

    默认从环境变量 MCP_RAG_URL 读取地址，否则用 DEFAULT_MCP_URL。
    这里假设返回结构类似用户给出的 JSON 示例。
    """
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
    """
    将复杂的 JSON（带 choices/context 等）转换成前端友好的结构：
    - answer: 直接展示给用户的回复文字
    - explanation: 用自然语言说明“我做了什么”
    - related_*: 相关内容数量/ID 列表，方便前端画图
    """
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

    # 这里不做复杂 NLP，只是做一点点清洗，让文本更易读
    cleaned_text = text.replace("\\n", "\n").strip()

    explanation = (
        "下面是根据知识库为你整理出的回答，我参考了多条相关内容，并尽量用通俗的语言说明。"
        "如果你觉得不准确，可以在问题里补充更多背景信息。"
    )

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
    """
    前端唯一需要调用的 RAG 接口：
    - 入参：自然语言问题
    - 出参：结构化、易懂的答案 + 相关条目数量（方便画图）
    """
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
    """健康检查，前端或监控可用"""
    return {"status": "ok"}

