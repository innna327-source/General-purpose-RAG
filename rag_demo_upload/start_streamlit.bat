@echo off
echo ========================================
echo   RAG系统 Streamlit 前端启动脚本
echo ========================================
echo.

echo [1/3] 检查依赖...
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo [!] 未检测到 streamlit，正在安装依赖...
    pip install streamlit plotly pandas
    if errorlevel 1 (
        echo [X] 依赖安装失败，请手动执行: pip install streamlit plotly pandas
        pause
        exit /b 1
    )
) else (
    echo [√] Streamlit 已安装
)

echo.
echo [2/3] 检查数据文件...
if not exist "output\test_report.json" (
    echo [!] 警告: 未找到 output\test_report.json
)
if not exist "index\*.chunks.jsonl" (
    echo [!] 警告: 未找到 chunks 数据文件
)

echo.
echo [3/3] 启动 Streamlit 应用...
echo.
echo ========================================
echo   应用将在浏览器中自动打开
echo   本地访问: http://localhost:8501
echo   按 Ctrl+C 停止服务
echo ========================================
echo.

streamlit run streamlit_app.py

pause
