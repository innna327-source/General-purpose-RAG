from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict

from flask import Flask, jsonify, request

from config.settings import SETTINGS
from mcp.handler import MCPHandler
from mcp.protocol import ERROR_RATE_LIMIT, make_error
from utils.paths import ensure_runtime_dirs


def create_app(handler: MCPHandler) -> Flask:
    ensure_runtime_dirs()
    app = Flask(__name__)

    # 简易 5 QPS 限流：1 秒时间窗
    qps_limit = 5
    window_sec = 1.0
    recent = deque()

    def rate_limited() -> bool:
        now = time.time()
        while recent and now - recent[0] > window_sec:
            recent.popleft()
        if len(recent) >= qps_limit:
            return True
        recent.append(now)
        return False

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"}), 200

    @app.get("/v1/models")
    def models():
        return (
            jsonify(
                {
                    "service": SETTINGS.mcp_service_name,
                    "protocol_version": SETTINGS.mcp_protocol_version,
                    "model_type": SETTINGS.mcp_model_type,
                    "capabilities": SETTINGS.mcp_capabilities,
                }
            ),
            200,
        )

    @app.post("/v1/completions")
    def completions():
        if rate_limited():
            return jsonify(make_error(ERROR_RATE_LIMIT, "Rate limit exceeded")), 429
        payload: Dict[str, Any] = request.get_json(silent=True) or {}
        resp, status = handler.handle_completions(payload)
        return jsonify(resp), status

    return app


def run_server(handler: MCPHandler, host: str, port: int) -> None:
    app = create_app(handler)
    # 单进程单线程约束：禁用 reloader，禁用 threaded
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=False)

