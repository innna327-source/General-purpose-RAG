"""
生成评估模块

评估指标：
- Faithfulness: 答案忠实度 - 答案是否基于检索内容生成
- Answer Match: 答案是否匹配文档中的关键信息（仅对 has_answer_in_doc=True 的查询评估）
- No-Answer Detection: 对于文档无答案的问题，LLM是否正确拒绝回答

核心思想：
- 检索分数高 ≠ 答案存在
- 需要验证生成答案是否真正使用了检索内容
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from generation.llm import generate_answer
from generation.constants import NO_ANSWER_PHRASES
from config.settings import SETTINGS


class GenerationEvaluator:
    """生成答案评估器"""

    def __init__(
        self,
        generation_model: str = None,
        api_key: str = None,
        base_url: str = None,
    ):
        self.model = generation_model or SETTINGS.generation_llm_model
        self.api_key = api_key or SETTINGS.llm_api_key
        self.base_url = base_url or SETTINGS.llm_base_url

    def _detect_no_answer(self, response: str) -> bool:
        """检测LLM是否返回了"无法回答"类型的响应"""
        for phrase in NO_ANSWER_PHRASES:
            if phrase in response:
                return True
        return False

    def evaluate_single(
        self,
        query: str,
        context_text: str,
        has_answer_in_doc: bool = None,
        answer_key_points: list[str] = None,
    ) -> dict:
        """
        评估单个查询的生成效果

        参数：
        - query: 用户问题
        - context_text: 检索到的上下文
        - has_answer_in_doc: 文档中是否有答案（标注）
        - answer_key_points: 答案应包含的关键点（标注）

        返回：
        - generated_answer: LLM生成的答案
        - is_no_answer_response: LLM是否返回了"无法回答"
        - faithfulness_correct: 忠实度是否正确
          - 如果 has_answer_in_doc=True 且 LLM返回了答案 → 正确
          - 如果 has_answer_in_doc=False 且 LLM说"无法回答" → 正确
          - 反之则错误
        """
        # 生成答案
        answer = generate_answer(
            query=query,
            context_text=context_text,
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
        )

        is_no_answer = self._detect_no_answer(answer)

        # 判断忠实度正确性
        faithfulness_correct = None
        if has_answer_in_doc is not None:
            if has_answer_in_doc:
                # 文档有答案，LLM应该回答而不是拒绝
                faithfulness_correct = not is_no_answer
            else:
                # 文档无答案，LLM应该拒绝回答
                faithfulness_correct = is_no_answer

        # 检查关键点匹配（如果有标注）
        key_points_matched = []
        if answer_key_points and not is_no_answer:
            for point in answer_key_points:
                if point.lower() in answer.lower():
                    key_points_matched.append(point)

        return {
            "generated_answer": answer,
            "is_no_answer_response": is_no_answer,
            "has_answer_in_doc": has_answer_in_doc,
            "faithfulness_correct": faithfulness_correct,
            "answer_key_points": answer_key_points,
            "key_points_matched": key_points_matched,
            "key_points_match_rate": (
                len(key_points_matched) / len(answer_key_points)
                if answer_key_points else None
            ),
        }

    def evaluate_batch(
        self,
        test_queries: list[dict],
        retriever: "HybridRetriever",
        top_k: int = 5,
    ) -> dict:
        """
        批量评估生成效果

        输入格式：
        [
            {
                "query": "问题文本",
                "expected_keywords": ["关键词"],
                "has_answer_in_doc": true/false,  # 必填！标注文档是否有答案
                "answer_key_points": ["关键点1"],  # 可选
            }
        ]
        """
        details = []
        total_with_answer = 0
        total_without_answer = 0
        correct_with_answer = 0
        correct_without_answer = 0
        no_answer_detected_count = 0

        for item in test_queries:
            query = item["query"]
            has_answer_in_doc = item.get("has_answer_in_doc")
            answer_key_points = item.get("answer_key_points", [])

            # 检索
            results, _ = retriever.hybrid_retrieve(query, top_k=top_k, return_debug_info=False)
            context_text = "\n\n---\n\n".join([r["text"] for r in results])

            # 生成评估
            result = self.evaluate_single(
                query=query,
                context_text=context_text,
                has_answer_in_doc=has_answer_in_doc,
                answer_key_points=answer_key_points,
            )

            if result["is_no_answer_response"]:
                no_answer_detected_count += 1

            # 统计正确率
            if has_answer_in_doc is not None:
                if has_answer_in_doc:
                    total_with_answer += 1
                    if result["faithfulness_correct"]:
                        correct_with_answer += 1
                else:
                    total_without_answer += 1
                    if result["faithfulness_correct"]:
                        correct_without_answer += 1

            details.append({
                "query": query,
                "has_answer_in_doc": has_answer_in_doc,
                "is_no_answer_response": result["is_no_answer_response"],
                "faithfulness_correct": result["faithfulness_correct"],
                "key_points_matched": result["key_points_matched"],
                "key_points_match_rate": result["key_points_match_rate"],
                "generated_answer": result["generated_answer"],
            })

        # 计算汇总指标
        faithfulness_for_answer = correct_with_answer / total_with_answer if total_with_answer > 0 else None
        faithfulness_for_no_answer = correct_without_answer / total_without_answer if total_without_answer > 0 else None

        # 总体忠实度（两种情况都正确才算正确）
        total_evaluated = total_with_answer + total_without_answer
        total_correct = correct_with_answer + correct_without_answer
        overall_faithfulness = total_correct / total_evaluated if total_evaluated > 0 else None

        return {
            "evaluation_type": "generation",
            "total_queries": len(test_queries),
            "queries_with_answer_in_doc": total_with_answer,
            "queries_without_answer_in_doc": total_without_answer,
            "no_answer_detected_count": no_answer_detected_count,
            "faithfulness_for_answer_queries": faithfulness_for_answer,  # 有答案问题的忠实度
            "faithfulness_for_no_answer_queries": faithfulness_for_no_answer,  # 无答案问题的忠实度
            "overall_faithfulness": overall_faithfulness,  # 总体忠实度
            "details": details,
        }

    def save_report(self, report: dict, out_path: Path) -> None:
        """保存评估报告"""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)