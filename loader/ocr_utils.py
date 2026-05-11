from __future__ import annotations

from typing import Optional


def ocr_pdf_pages_to_text(pdf_path: str) -> str:
    """
    OCR 为可选能力：
    - 若 Tesseract 不可用，返回空字符串，由调用方决定是否抛错
    """
    try:
        import fitz  # pymupdf
        import pytesseract
        from PIL import Image
    except Exception:
        return ""

    # 检测 tesseract 是否可用
    try:
        _ = pytesseract.get_tesseract_version()
    except Exception:
        return ""

    doc = fitz.open(pdf_path)
    parts: list[str] = []
    for page in doc:
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        try:
            txt = pytesseract.image_to_string(img, lang="chi_sim+eng")
        except Exception:
            txt = ""
        if txt:
            parts.append(txt)
    doc.close()
    return "\n".join(parts)

