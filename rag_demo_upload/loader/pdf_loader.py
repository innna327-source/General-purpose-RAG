from __future__ import annotations

import warnings

import fitz  # pymupdf

from loader.base_loader import BaseLoader
from loader.ocr_utils import ocr_pdf_pages_to_text


class PDFLoader(BaseLoader):
    def load(self, file_path: str) -> str:
        doc = fitz.open(file_path)
        parts: list[str] = []
        for page in doc:
            try:
                parts.append(page.get_text("text"))
            except Exception:
                parts.append("")
        doc.close()
        text = "\n".join(parts)

        if len(text.strip()) > 100:
            return text

        # fallback OCR
        ocr_text = ocr_pdf_pages_to_text(file_path)
        if not ocr_text.strip():
            warnings.warn(
                "PDF 文本提取不足且 OCR 不可用/无结果，将抛出异常。",
                RuntimeWarning,
            )
        final_text = (text + "\n" + ocr_text).strip()
        if len(final_text.strip()) <= 100:
            raise ValueError("PDF 无可提取文本内容")
        return final_text

