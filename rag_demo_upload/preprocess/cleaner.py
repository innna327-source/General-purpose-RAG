from __future__ import annotations

import re
import unicodedata
from typing import List


_PUNCT_TO_REMOVE = r"""\.\,\!\?\;\:\，。\！？；：“”‘’（）\(\)【】\[\]<>《》"""


def normalize_for_match(text: str) -> str:
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.strip().lower()
    t = re.sub(rf"[{_PUNCT_TO_REMOVE}]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


_DEFAULT_CHAR_PATTERN = re.compile(
    r"[^\x09\x0A\x0D\x20-\x7E"   # ASCII 可打印字符 + 基本控制符
    r"\u4e00-\u9fff"               # CJK 基本汉字
    r"\u3400-\u4dbf"               # CJK 扩展 A（繁体/生僻字）
    r"\uff00-\uffef"               # 全角符号（含货币、数学）
    r"\u2000-\u206f"               # 常用标点
    r"]"
)
"""
默认字符过滤正则：覆盖中英文 + 繁体 + 全角符号。
原版只保留 CJK 基本区（\u4e00-\u9fff），会误删：
  - 繁体字（法律/金融文书常见）
  - 全角货币符号 ￥€£
  - 数学/特殊标点
如需支持日文/韩文，在调用 clean_to_paragraphs 时传入自定义 extra_char_ranges。
"""


def _clean_basic(text: str, char_pattern: re.Pattern = _DEFAULT_CHAR_PATTERN) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    t = t.replace("\x00", " ")
    t = re.sub(r"[^\S\r\n]+", " ", t)  # 多余空格（保留换行）
    t = re.sub(r"\n{3,}", "\n\n", t)  # 连续换行
    t = char_pattern.sub("", t)        # 过滤非法字符（白名单可由调用方覆盖）
    return t.strip()


def _split_long_sentence(s: str, max_len: int = 200) -> List[str]:
    s = s.strip()
    if len(s) <= max_len:
        return [s] if s else []
    parts = re.split(r"([。！？；;.!?])", s)
    merged: list[str] = []
    buf = ""
    for i in range(0, len(parts), 2):
        piece = parts[i].strip()
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        seg = (piece + punct).strip()
        if not seg:
            continue
        if len(buf) + len(seg) + 1 <= max_len:
            buf = (buf + " " + seg).strip()
        else:
            if buf:
                merged.append(buf)
            buf = seg
    if buf:
        merged.append(buf)
    return merged


def clean_to_paragraphs(
    text: str,
    char_pattern: re.Pattern = _DEFAULT_CHAR_PATTERN,
) -> List[str]:
    """
    输入：str
    输出：List[str] 段落列表

    char_pattern：字符过滤白名单正则，默认保留中英文+繁体+全角符号。
                  如需保留日文/韩文/特殊符号，传入自定义 re.Pattern。
    """
    t = _clean_basic(text, char_pattern=char_pattern)
    # 段落结构规整：以空行分段
    raw_paras = [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]
    paras: list[str] = []
    for p in raw_paras:
        p = re.sub(r"\s*\n\s*", " ", p).strip()
        paras.extend(_split_long_sentence(p))
    # 再次去掉空白
    return [p for p in (pp.strip() for pp in paras) if p]

