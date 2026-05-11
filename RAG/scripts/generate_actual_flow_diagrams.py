from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_FLOW = ROOT / "rag_actual_flow_text.png"
OUT_LAYER = ROOT / "rag_actual_layer_arch.png"

BG = "#0b0b0b"
FG = "#d7d7d7"
MUTED = "#a9a9a9"
LINE = "#cfcfcf"
ACCENT = "#8fc7ff"
GOOD = "#9ad29a"
WARN = "#e7c07a"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        r"C:\Windows\Fonts\NotoSansSC-VF.ttf",
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


TITLE = font(36, True)
TEXT = font(30)
SMALL = font(24)
BOX_TITLE = font(33, True)
BOX_TEXT = font(29)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, fill: str = FG,
         fnt: ImageFont.ImageFont = TEXT) -> None:
    draw.text(xy, value, fill=fill, font=fnt)


def multiline(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, fill: str = FG,
              fnt: ImageFont.ImageFont = TEXT, spacing: int = 12) -> None:
    draw.multiline_text(xy, value, fill=fill, font=fnt, spacing=spacing)


def arrow_down(draw: ImageDraw.ImageDraw, x: int, y1: int, y2: int, fill: str = LINE) -> None:
    draw.line((x, y1, x, y2), fill=fill, width=3)
    draw.line((x, y2, x - 9, y2 - 13), fill=fill, width=3)
    draw.line((x, y2, x + 9, y2 - 13), fill=fill, width=3)


def box(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], title: str, body: str,
        side: str | None = None) -> None:
    x1, y1, x2, y2 = xy
    draw.rectangle(xy, outline=LINE, width=3)
    text(draw, (x1 + 40, y1 + 30), title, fnt=BOX_TITLE)
    multiline(draw, (x1 + 40, y1 + 82), body, fnt=BOX_TEXT, spacing=10)
    if side:
        multiline(draw, (x2 + 48, y1 + 34), side, fnt=BOX_TEXT, spacing=10)


def generate_flow() -> None:
    img = Image.new("RGB", (1900, 1420), BG)
    draw = ImageDraw.Draw(img)

    lines = [
        ("PDF 文件", FG),
        ("↓", LINE),
        ("main.py（入口编排：test / mcp / eval / rollback）", FG),
        ("↓", LINE),
        ("preprocess/hash_checker.py（SHA256 增量检查）", FG),
        ("├─ 命中且索引完整 → retrieval/factory.py 加载现有索引", GOOD),
        ("└─ 未命中 / 索引缺失 → 全量构建：", WARN),
        ("   loader/factory.py → loader/pdf_loader.py（PyMuPDF 解析 / OCR 回退 → 文本）", FG),
        ("   preprocess/cleaner.py（清洗 → 段落）", FG),
        ("   preprocess/deduplicator.py（段落去重，阈值 0.85）", FG),
        ("   chunk/hierarchical_chunk.py（层级分块 → 父块 + 子块）", FG),
        ("   graph/semantic_graph.py（语义图谱：nodes / edges / entity_chunks）", FG),
        ("   retrieval/bm25.py（保存 chunks.jsonl + BM25 索引）", FG),
        ("   retrieval/vector_store.py（FAISS 向量索引）", FG),
        ("   preprocess/hash_checker.py（record_hash + version_log）", FG),
        ("↓", LINE),
        ("retrieval/factory.py（统一加载 BM25 + FAISS + 图谱 + 重排配置）", FG),
        ("↓", LINE),
        ("retrieval/hybrid.py（在线混合检索）", FG),
        ("   ├─ BM25：扩展 query 后关键词召回", FG),
        ("   ├─ FAISS：原始 query 向量召回", FG),
        ("   ├─ 图谱：多跳扩展 + entity_chunks 召回", FG),
        ("   ├─ 融合：min-max 归一化 → 0.7/0.3 加权 → graph boost", ACCENT),
        ("   └─ 重排：Cross-Encoder BAAI/bge-reranker-v2-m3，对 top50 重排", ACCENT),
        ("↓", LINE),
        ("generation/llm.py（构造 Prompt + 调用 LLM） ← generation/constants.py（共享拒答常量）", FG),
        ("↓", LINE),
        ("streamlit_app.py / MCP HTTP 接口（白盒 UI / 外部调用）", FG),
        ("↑", LINE),
        ("utils/queries.py（测试查询）  utils/index_paths.py（索引路径）  utils/logger.py（日志）", MUTED),
    ]

    y = 28
    for value, color in lines:
        fnt = TEXT
        if value.startswith("retrieval/hybrid.py") or value.startswith("preprocess/hash_checker.py"):
            fnt = font(32, True)
        text(draw, (28, y), value, color, fnt)
        y += 45 if value in {"↓", "↑"} else 47

    img.save(OUT_FLOW)


def generate_layer() -> None:
    img = Image.new("RGB", (1800, 1540), BG)
    draw = ImageDraw.Draw(img)

    box(
        draw,
        (32, 36, 1040, 206),
        "main.py — 入口编排层",
        '决定 "先做什么、后做什么"\nmode=test / mcp，支持 eval 和 rollback',
        "流程控制者\n哈希检查后选择\n加载旧索引或全量构建",
    )
    arrow_down(draw, 315, 208, 330)

    box(
        draw,
        (32, 360, 1040, 730),
        "核心业务层",
        "- loader/（PDF 解析，OCR 回退）\n"
        "- preprocess/（hash 增量、清洗、去重）\n"
        "- chunk/（层级分块：父块 + 子块）\n"
        "- graph/（语义图谱、多跳扩展、实体召回）\n"
        "- retrieval/（BM25 + FAISS + min-max 融合\n"
        "  + Cross-Encoder 重排）\n"
        "- generation/（Prompt 构造 + LLM 生成）",
        "流程本身\n每个目录承担一段\n可独立测试的复杂逻辑",
    )
    arrow_down(draw, 315, 732, 820)

    box(
        draw,
        (32, 850, 1040, 1130),
        "服务与评估层",
        "- streamlit_app.py（白盒可视化 UI）\n"
        "- MCP HTTP 接口 + mcp/handler.py（外部 Agent 调用）\n"
        "- evaluation/（检索评估、生成评估）",
        "对外入口\n把核心流程包装成\nAPI / UI / 测试报告",
    )
    arrow_down(draw, 315, 1132, 1210)

    box(
        draw,
        (32, 1230, 1040, 1480),
        "工具与配置层",
        "config/settings.py\n"
        "utils/index_paths.py  ·  utils/queries.py\n"
        "utils/logger.py  ·  utils/paths.py",
        '真正的"工具"\n被多个模块复用',
    )

    img.save(OUT_LAYER)


if __name__ == "__main__":
    generate_flow()
    generate_layer()
    print(f"saved: {OUT_FLOW}")
    print(f"saved: {OUT_LAYER}")
