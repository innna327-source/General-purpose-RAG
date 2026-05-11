"""
统一评估脚本

运行方式：
python run_evaluation.py --file your.pdf

会同时生成：
1. 检索评估报告 (retrieval_report.json)
2. 生成评估报告 (generation_report.json)
3. 综合评估报告 (full_report.json)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config.settings import SETTINGS
from evaluation import GenerationEvaluator, RetrievalEvaluator
from retrieval.factory import load_retriever
from utils import logger
from utils.queries import load_test_queries


def run_evaluation(file_hash: str, output_dir: Path) -> dict:
    """运行完整评估"""
    logger.info(f"加载检索器 hash={file_hash}")
    retriever = load_retriever(file_hash)

    test_queries = load_test_queries()
    logger.info(f"加载测试查询 {len(test_queries)} 条")

    # 1. 检索评估
    logger.info("=" * 50)
    logger.info("【检索评估】评估召回质量...")
    retrieval_evaluator = RetrievalEvaluator(retriever)
    retrieval_report = retrieval_evaluator.evaluate_batch(test_queries)
    retrieval_evaluator.save_report(retrieval_report, output_dir / "retrieval_report.json")
    logger.info(f"Recall@5: {retrieval_report['recall_at_5']:.2%}")
    logger.info(f"MRR: {retrieval_report['mrr']:.3f}")
    if retrieval_report['answer_recall'] is not None:
        logger.info(f"有答案问题召回率: {retrieval_report['answer_recall']:.2%}")

    # 2. 生成评估
    logger.info("=" * 50)
    logger.info("【生成评估】评估答案忠实度...")
    generation_evaluator = GenerationEvaluator()
    generation_report = generation_evaluator.evaluate_batch(test_queries, retriever)
    generation_evaluator.save_report(generation_report, output_dir / "generation_report.json")

    if generation_report['overall_faithfulness'] is not None:
        logger.info(f"总体忠实度: {generation_report['overall_faithfulness']:.2%}")
    if generation_report['faithfulness_for_answer_queries'] is not None:
        logger.info(f"有答案问题忠实度: {generation_report['faithfulness_for_answer_queries']:.2%}")
    if generation_report['faithfulness_for_no_answer_queries'] is not None:
        logger.info(f"无答案问题忠实度: {generation_report['faithfulness_for_no_answer_queries']:.2%}")

    # 3. 综合报告
    full_report = {
        "summary": {
            "total_queries": len(test_queries),
            "queries_with_answer_in_doc": retrieval_report["queries_with_answer_in_doc"],
            "retrieval": {
                "recall_at_5": retrieval_report["recall_at_5"],
                "mrr": retrieval_report["mrr"],
                "answer_recall": retrieval_report["answer_recall"],
            },
            "generation": {
                "overall_faithfulness": generation_report["overall_faithfulness"],
                "faithfulness_for_answer_queries": generation_report["faithfulness_for_answer_queries"],
                "faithfulness_for_no_answer_queries": generation_report["faithfulness_for_no_answer_queries"],
            },
        },
        "retrieval_report": retrieval_report,
        "generation_report": generation_report,
    }

    full_report_path = output_dir / "full_report.json"
    with open(full_report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
    logger.info(f"综合报告已保存: {full_report_path}")

    return full_report


def main():
    parser = argparse.ArgumentParser(description="RAG系统评估")
    parser.add_argument("--file", type=str, required=True, help="PDF文件名")
    parser.add_argument("--index-hash", type=str, default=None, help="索引hash（可选）")
    args = parser.parse_args()

    # 获取 file_hash
    if args.index_hash:
        file_hash = args.index_hash
    else:
        db_path = SETTINGS.hash_db_path
        if not db_path.exists():
            raise ValueError("未找到 hash_db.json，请先运行索引构建")
        with open(db_path, "r", encoding="utf-8") as f:
            hash_db = json.load(f)

        # 查找文件对应的hash
        for h, meta in hash_db.items():
            if meta.get("file_name") == args.file:
                file_hash = h
                break
        else:
            raise ValueError(f"未找到文件 {args.file} 的索引记录")

    output_dir = SETTINGS.output_dir
    report = run_evaluation(file_hash, output_dir)

    # 打印摘要
    print("\n" + "=" * 60)
    print("📊 RAG 系统评估报告摘要")
    print("=" * 60)
    print(f"总查询数: {report['summary']['total_queries']}")
    print(f"有答案问题数: {report['summary']['queries_with_answer_in_doc']}")
    print("-" * 60)
    print("【检索评估】")
    print(f"  Recall@5: {report['summary']['retrieval']['recall_at_5']:.2%}")
    print(f"  MRR: {report['summary']['retrieval']['mrr']:.3f}")
    print("-" * 60)
    print("【生成评估】")
    gen = report['summary']['generation']
    if gen['overall_faithfulness'] is not None:
        print(f"  总体忠实度: {gen['overall_faithfulness']:.2%}")
    if gen['faithfulness_for_answer_queries'] is not None:
        print(f"  有答案问题忠实度: {gen['faithfulness_for_answer_queries']:.2%}")
    if gen['faithfulness_for_no_answer_queries'] is not None:
        print(f"  无答案问题忠实度: {gen['faithfulness_for_no_answer_queries']:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
