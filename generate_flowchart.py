import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

matplotlib.rcParams["font.family"] = "Microsoft YaHei"
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 配色方案（莫兰迪风格）──────────────────────────────
BG        = "#F4F1ED"       # 暖米色背景
CARD_BG   = "#FFFFFF"       # 卡片底色
DIVIDER   = "#D9CFC4"       # 分割线

C_INPUT   = "#B5C9D8"       # 输入节点：雾蓝
C_PARSE   = "#A8C5A0"       # 解析层：莫兰迪绿
C_PREPROC = "#C4B3D0"       # 预处理：薰衣草紫
C_CHUNK   = "#E8C9A0"       # 分块：暖杏色
C_INDEX   = "#9DC4C4"       # 索引：青绿
C_RETRIEV = "#B8A9C9"       # 检索：柔紫
C_FUSION  = "#E8B4A0"       # 融合：珊瑚
C_LLM     = "#C9A0A0"       # LLM：玫瑰红
C_OUTPUT  = "#A0C9B0"       # 输出：薄荷绿
C_STORAGE = "#D4C5A9"       # 存储：卡其

TEXT_DARK  = "#2C2C2C"
TEXT_MID   = "#555555"
TEXT_LIGHT = "#FFFFFF"
ARROW_C    = "#8A8A8A"

fig, ax = plt.subplots(figsize=(16, 22))
ax.set_xlim(0, 16)
ax.set_ylim(0, 22)
ax.axis("off")
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# ── 工具函数 ─────────────────────────────────────────
def card(ax, x, y, w, h, title, subtitle="", fill=C_INDEX, text_color=TEXT_DARK, radius=0.25):
    shadow = FancyBboxPatch((x - w/2 + 0.06, y - h/2 - 0.06), w, h,
                             boxstyle=f"round,pad=0.0,rounding_size={radius}",
                             linewidth=0, facecolor="#C0B8B0", alpha=0.35, zorder=2)
    ax.add_patch(shadow)
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle=f"round,pad=0.0,rounding_size={radius}",
                           linewidth=1.2, edgecolor="#CCCCCC", facecolor=fill, zorder=3)
    ax.add_patch(rect)
    if subtitle:
        ax.text(x, y + 0.12, title, ha="center", va="center", fontsize=9.5,
                color=text_color, fontweight="bold", zorder=4)
        ax.text(x, y - 0.22, subtitle, ha="center", va="center", fontsize=7.5,
                color=TEXT_MID, zorder=4)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=9.5,
                color=text_color, fontweight="bold", zorder=4)

def arr(ax, x1, y1, x2, y2, color=ARROW_C, lw=1.6):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=14),
                zorder=5)

def section_label(ax, x, y, text, color):
    ax.text(x, y, text, ha="center", va="center", fontsize=11,
            fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=CARD_BG,
                      edgecolor=color, linewidth=1.5, alpha=0.9))

def divider(ax, x, y, w, color=DIVIDER):
    ax.plot([x - w/2, x + w/2], [y, y], color=color, lw=1, ls="--", zorder=2)

# ══════════════════════════════════════════════════════
# 标题
# ══════════════════════════════════════════════════════
ax.text(8, 21.5, "Deep RAG Demo — 系统数据流", ha="center", va="center",
        fontsize=16, fontweight="bold", color=TEXT_DARK,
        path_effects=[pe.withStroke(linewidth=3, foreground=BG)])
ax.text(8, 21.0, "从 PDF 上传到生成回答的完整处理路径", ha="center", va="center",
        fontsize=10, color=TEXT_MID)

# ── 横向分割线 ────────────────────────────────────────
ax.plot([0.5, 7.5], [10.6, 10.6], color=DIVIDER, lw=1.5, ls="--", zorder=2)
ax.plot([8.5, 15.5], [10.6, 10.6], color=DIVIDER, lw=1.5, ls="--", zorder=2)

# ══════════════════════════════════════════════════════
# 左侧：离线索引阶段
# ══════════════════════════════════════════════════════
section_label(ax, 4, 20.4, "① 离线索引阶段  |  PDF 上传时触发", "#5B7FA6")

LX = 4.0   # 左侧 x 中心
W  = 5.2
H  = 0.82

nodes_left = [
    (LX, 19.5, "📄  PDF 原始文件",              "",                                      C_INPUT,   TEXT_DARK),
    (LX, 18.3, "hash_checker.py",              "计算 SHA256，查询已索引文件数据库",        "#D4C5A9",  TEXT_DARK),
    (LX, 17.1, "pdf_loader.py",                "pdfplumber 解析 / Tesseract OCR 回退",  C_PARSE,   TEXT_DARK),
    (LX, 15.9, "cleaner.py",                   "清洗噪声字符、统一文本格式",               C_PREPROC, TEXT_DARK),
    (LX, 14.7, "deduplicator.py",              "段落相似度去重（阈值 0.85）",              C_PREPROC, TEXT_DARK),
    (LX, 13.5, "hierarchical_chunk.py",        "父块 + 子块两层分块 → 输出 JSONL",         C_CHUNK,   TEXT_DARK),
]

for x, y, title, sub, fill, tc in nodes_left:
    card(ax, x, y, W, H, title, sub, fill, tc)

# 主链箭头
for i in range(len(nodes_left) - 1):
    _, y1, *_ = nodes_left[i]
    _, y2, *_ = nodes_left[i+1]
    arr(ax, LX, y1 - H/2, LX, y2 + H/2)

# Hash 命中 → 跳过处理，旁路箭头
SKIP_X = LX + W/2 + 0.3
skip_top_y = 18.3
skip_bot_y = 11.0   # 指向持久化节点旁
ax.annotate("", xy=(SKIP_X, skip_bot_y), xytext=(SKIP_X, skip_top_y),
            arrowprops=dict(arrowstyle="-|>", color="#C9855A", lw=1.6,
                            connectionstyle="arc3,rad=0.0", mutation_scale=14),
            zorder=5)
ax.plot([LX + W/2, SKIP_X], [skip_top_y, skip_top_y], color="#C9855A", lw=1.6, zorder=5)
ax.plot([SKIP_X, LX + W/2 + 0.05], [skip_bot_y, skip_bot_y], color="#C9855A", lw=1.6, zorder=5)
ax.text(SKIP_X + 0.15, (skip_top_y + skip_bot_y) / 2,
        "哈希命中\n跳过处理", va="center", ha="left", fontsize=7.5,
        color="#C9855A", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF5EE",
                  edgecolor="#C9855A", linewidth=1.0, alpha=0.9))

# 三叉分支
forks = [
    (1.6,  12.2, "vector_store.py",  "FAISS 向量索引", C_INDEX),
    (4.0,  12.2, "bm25.py",          "BM25 倒排索引",  C_INDEX),
    (6.4,  12.2, "semantic_graph.py","语义关系图谱",   C_INDEX),
]
FW, FH = 3.2, 0.82
for x, y, title, sub, fill in forks:
    card(ax, x, y, FW, FH, title, sub, fill)
    arr(ax, LX, nodes_left[-1][1] - H/2, x, y + FH/2)

# 持久化存储
card(ax, LX, 11.0, 6.8, 0.72,
     "💾  持久化索引文件",
     "faiss.index  ·  bm25.pkl  ·  graph.pkl",
     C_STORAGE, TEXT_DARK)
for x, y, *_ in forks:
    arr(ax, x, y - FH/2, LX, 11.0 + 0.36)

# ══════════════════════════════════════════════════════
# 右侧：在线检索阶段
# ══════════════════════════════════════════════════════
section_label(ax, 12, 20.4, "② 在线检索阶段  |  用户提问时触发", "#5B7FA6")

RX = 12.0
nodes_right_top = [
    (RX, 19.5, "🙋  用户输入问题",  "",                       C_INPUT,   TEXT_DARK),
]
card(ax, RX, 19.5, W, H, "🙋  用户输入问题", "", C_INPUT, TEXT_DARK)

# 双路检索
branches = [
    (10.0, 18.3, "bm25.py",          "关键词检索 + 查询扩展",  C_RETRIEV),
    (14.0, 18.3, "vector_store.py",  "语义向量检索",           C_RETRIEV),
]
BW, BH = 3.6, 0.82
for x, y, title, sub, fill in branches:
    card(ax, x, y, BW, BH, title, sub, fill)
    arr(ax, RX, 19.5 - H/2, x, y + BH/2)

# RRF 融合
card(ax, RX, 17.1, W, H, "hybrid.py", "RRF 融合  ·  BM25 : 向量 = 0.7 : 0.3", C_FUSION)
for x, y, *_ in branches:
    arr(ax, x, y - BH/2, RX, 17.1 + H/2)

# 图谱扩展
card(ax, RX, 15.9, W, H, "semantic_graph.py", "图谱节点扩展，补充关联文本块", C_RETRIEV)
arr(ax, RX, 17.1 - H/2, RX, 15.9 + H/2)

# 子块→父块
card(ax, RX, 14.7, W, H, "子块命中 → 返回父块", "保语义完整性，送入 LLM", C_CHUNK)
arr(ax, RX, 15.9 - H/2, RX, 14.7 + H/2)

# LLM
card(ax, RX, 13.2, W, H, "llm.py  ·  GLM-4", "构造 Prompt → 调用大模型", C_LLM, TEXT_DARK)
arr(ax, RX, 14.7 - H/2, RX, 13.2 + H/2)

# 输出
card(ax, RX, 11.8, W, H, "✅  最终回答", "自然语言，有据可查", C_OUTPUT, TEXT_DARK)
arr(ax, RX, 13.2 - H/2, RX, 11.8 + H/2)

# 返回接口 / 可视化
card(ax, RX, 10.6, W, H, "MCP HTTP接口  /  Streamlit", "外部调用或白盒展示", C_STORAGE, TEXT_DARK)
arr(ax, RX, 11.8 - H/2, RX, 10.6 + H/2)

# ══════════════════════════════════════════════════════
# 底部：图例
# ══════════════════════════════════════════════════════
legend_data = [
    (C_INPUT,   "输入节点"),
    (C_PARSE,   "解析模块"),
    (C_PREPROC, "预处理模块"),
    (C_CHUNK,   "分块 / 召回"),
    (C_INDEX,   "索引构建"),
    (C_RETRIEV, "检索 / 图谱"),
    (C_FUSION,  "融合模块"),
    (C_LLM,     "LLM 生成"),
    (C_OUTPUT,  "输出节点"),
    (C_STORAGE, "存储 / 接口"),
]
lx_start = 0.8
for i, (color, label) in enumerate(legend_data):
    lx = lx_start + i * 1.53
    ly = 9.8
    rect = FancyBboxPatch((lx, ly), 0.28, 0.28,
                           boxstyle="round,pad=0.03",
                           linewidth=0.8, edgecolor="#BBBBBB", facecolor=color, zorder=3)
    ax.add_patch(rect)
    ax.text(lx + 0.38, ly + 0.14, label, va="center", fontsize=7, color=TEXT_MID)

ax.text(8, 9.4, "模块颜色图例", ha="center", fontsize=8, color=TEXT_MID)

# ══════════════════════════════════════════════════════
# 保存
# ══════════════════════════════════════════════════════
plt.tight_layout(pad=0.5)
plt.savefig("C:/rag_demo/rag_dataflow.png", dpi=160, bbox_inches="tight",
            facecolor=BG)
print("saved: C:/rag_demo/rag_dataflow.png")
