from __future__ import annotations

import os
from typing import Optional

_SYSTEM_PROMPT = """\
你是一个严谨的文档问答助手。请严格依据下方【参考资料】中的内容回答用户问题。

规则：
1. 只使用参考资料中明确出现的信息，不得推断或补充资料外的内容。
2. 如果参考资料中没有足够信息来回答问题，直接回复"根据现有资料无法回答该问题"，不要编造答案。
3. 回答简洁，直接引用资料中的关键句，避免无关废话。
"""


def generate_answer(
    query: str,
    context_text: str,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> str:
    """
    调用 OpenAI 兼容接口根据检索到的 context_text 生成答案。
    若 api_key 为空则直接返回原始 context_text（降级）。
    """
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return context_text  # 无 API key，降级为直接返回检索结果

    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("未安装 openai 包，请运行：pip install openai")

    user_message = f"【参考资料】\n{context_text}\n\n【问题】\n{query}"

    client = OpenAI(api_key=key, base_url=base_url or None)
    response = client.chat.completions.create(
        model=model,
        max_tokens=512,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content
