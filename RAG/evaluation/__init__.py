"""
RAG 系统评估模块

区分两种评估维度：
1. 检索评估：衡量召回质量
2. 生成评估：衡量答案忠实度
"""

from evaluation.retrieval_eval import RetrievalEvaluator
from evaluation.generation_eval import GenerationEvaluator

__all__ = ["RetrievalEvaluator", "GenerationEvaluator"]