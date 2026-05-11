"""测试Streamlit应用环境"""
from pathlib import Path

print("=" * 50)
print("Streamlit应用环境检查")
print("=" * 50)

# 检查依赖
print("\n[1] 检查依赖包...")
deps_ok = True
try:
    import streamlit
    print("  - streamlit: OK")
except ImportError:
    print("  - streamlit: 未安装")
    deps_ok = False

try:
    import plotly
    print("  - plotly: OK")
except ImportError:
    print("  - plotly: 未安装")
    deps_ok = False

try:
    import pandas
    print("  - pandas: OK")
except ImportError:
    print("  - pandas: 未安装")
    deps_ok = False

# 检查数据文件
print("\n[2] 检查数据文件...")
chunks_file = Path("index/d1257ca2ac08ed674fe3315319dd64b423e51991f2b3932a5c8fc697a1da97a3.chunks.jsonl")
report_file = Path("output/test_report.json")
queries_file = Path("tests/test_queries1.json")

chunks_ok = chunks_file.exists()
report_ok = report_file.exists()
queries_ok = queries_file.exists()

print(f"  - Chunks数据: {'OK' if chunks_ok else '未找到'}")
print(f"  - 测试报告: {'OK' if report_ok else '未找到'}")
print(f"  - 测试查询: {'OK' if queries_ok else '未找到'}")

# 总结
print("\n" + "=" * 50)
if deps_ok and chunks_ok and report_ok and queries_ok:
    print("状态: 准备就绪!")
    print("\n启动命令:")
    print("  streamlit run streamlit_app.py")
    print("\n或双击运行:")
    print("  start_streamlit.bat")
else:
    print("状态: 需要配置")
    if not deps_ok:
        print("\n请安装依赖:")
        print("  pip install streamlit plotly pandas")
    if not (chunks_ok and report_ok and queries_ok):
        print("\n缺少数据文件，请先运行主程序生成数据")
print("=" * 50)
