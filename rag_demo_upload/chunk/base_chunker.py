from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ChunkResult:
    """
    所有分块器的统一输出格式。
    从 hierarchical_chunk 中提升到此处，作为模块公共数据契约，
    避免调用方依赖具体实现模块来获取类型定义。
    """
    chunks: List[dict]
    parent_child_map: Dict[str, List[str]]
    stats: dict


class BaseChunker(ABC):
    """
    所有分块策略的统一接口。

    输入：clean 后的段落列表（由 preprocess 层产出）
    输出：ChunkResult（chunks + parent_child_map + stats）

    约定：实现类只负责"怎么切"，不感知文档类型、领域知识、存储格式。
    领域相关的参数（标题正则、同义词等）由调用方通过构造函数注入，
    不得硬编码在实现类内部。
    """

    @abstractmethod
    def chunk(self, paragraphs: List[str]) -> ChunkResult:
        raise NotImplementedError
