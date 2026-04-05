@echo off
setlocal enabledelayedexpansion

echo [1/4] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
  echo pip install failed. Please check your Python environment.
  exit /b 1
)

echo [2/4] FAISS on Windows (recommended: conda-forge)
echo   conda install -c conda-forge faiss-cpu
echo   python -c "import faiss; print('faiss ok')"
echo.

echo [3/4] OCR (optional)
echo   - This demo uses pytesseract, but you must install Tesseract manually on Windows.
echo   - If Tesseract is not installed, OCR will be skipped automatically.
echo.

echo [4/4] Done.
echo You can run:
echo   python main.py --mode test --file your.pdf
echo   python main.py --mode test --file your.pdf --eval
echo   python main.py --mode mcp --index-hash ^<file_hash^>

endlocal

