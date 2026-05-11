"""
Streamlit前端应用 - RAG系统数据展示
展示数据处理大盘、检索质量跑分看板、召回链路透视表和分数对比柱状图
"""
import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# 页面配置
st.set_page_config(
    page_title="RAG系统数据展示",
    page_icon="📊",
    layout="wide"
)

# 标题
st.title("📊 RAG系统数据展示大盘")
st.markdown("---")

# 数据加载函数
@st.cache_data
def load_chunks_data():
    """加载chunks数据"""
    chunks_file = Path("index/d1257ca2ac08ed674fe3315319dd64b423e51991f2b3932a5c8fc697a1da97a3.chunks.jsonl")
    if not chunks_file.exists():
        return []
    
    chunks = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

@st.cache_data
def load_test_report():
    """加载测试报告"""
    report_file = Path("output/test_report.json")
    if not report_file.exists():
        return None
    
    with open(report_file, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def load_test_queries():
    """加载测试查询"""
    queries_file = Path("tests/test_queries.json")
    if not queries_file.exists():
        return []
    
    with open(queries_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
            return [{"query": q} for q in data]
        return data

# 加载数据
chunks_data = load_chunks_data()
test_report = load_test_report()
test_queries = load_test_queries()

# ==================== 1. 数据处理大盘 ====================
st.header("📦 数据处理大盘 (Data Processing Dashboard)")
st.markdown("**展示内容：** 使用大号数字卡片展示文档的分块统计信息（总父块数、总子块数、平均长度）")
st.markdown('**作用：** 让用户直观看到"一篇PDF到底被切成了多少块"')

if chunks_data:
    # 计算统计信息
    total_chunks = len(chunks_data)
    parent_ids = set(chunk['parent_id'] for chunk in chunks_data)
    total_parents = len(parent_ids)
    avg_length = sum(len(chunk['text']) for chunk in chunks_data) / total_chunks
    
    # 使用列布局展示卡片
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="总父块数",
            value=f"{total_parents}",
            help="文档被分割成的父级块数量"
        )
    
    with col2:
        st.metric(
            label="总子块数",
            value=f"{total_chunks}",
            help="所有子块的总数量"
        )
    
    with col3:
        st.metric(
            label="平均长度",
            value=f"{avg_length:.0f} 字符",
            help="每个块的平均字符长度"
        )
else:
    st.warning("未找到chunks数据文件")

st.markdown("---")

# ==================== 2. 检索质量跑分看板 ====================
st.header("🎯 检索质量跑分看板 (Evaluation Metrics)")
st.markdown("**展示内容：** 展示 test_report.json 中的 recall_at_5 和 mrr 核心指标")
st.markdown("**作用：** 证明你的检索算法是有效且高质量的")

if test_report:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="总查询数",
            value=test_report['total_queries'],
            help="测试集中的查询总数"
        )
    
    with col2:
        recall_value = test_report['recall_at_5']
        st.metric(
            label="Recall@5",
            value=f"{recall_value:.2%}",
            delta=f"{recall_value - 0.5:.2%}" if recall_value > 0.5 else None,
            help="前5个结果中命中相关文档的比例"
        )
    
    with col3:
        mrr_value = test_report['mrr']
        st.metric(
            label="MRR (平均倒数排名)",
            value=f"{mrr_value:.3f}",
            delta=f"{mrr_value - 0.5:.3f}" if mrr_value > 0.5 else None,
            help="衡量相关结果排名的平均质量"
        )
        
    # 如果有通过率数据，显示通过率
    if 'pass_rate' in test_report:
        st.markdown("---")
        st.subheader("🎯 召回准确率 (基于关键词匹配)")
        
        col_pr1, col_pr2, col_pr3 = st.columns(3)
        with col_pr1:
            st.metric(
                label="评估查询总数",
                value=test_report['total_queries']
            )
        with col_pr2:
            st.metric(
                label="成功召回查询数",
                value=test_report['passed_queries']
            )
        with col_pr3:
            pass_rate = test_report['pass_rate']
            st.metric(
                label="关键词召回率",
                value=f"{pass_rate:.2%}",
                help="Top 5 结果中包含预期关键词的查询比例"
            )
    
    # 详细结果表格
    st.subheader("📋 查询详细结果")
    details_df = pd.DataFrame(test_report['details'])
    details_df['命中排名'] = details_df['hit_rank'].apply(lambda x: f"第{x}名" if x > 0 else "未命中")
    details_df['倒数排名'] = details_df['reciprocal_rank'].apply(lambda x: f"{x:.3f}")
    
    if 'is_keyword_hit' in details_df.columns:
        details_df['关键词命中状态'] = details_df['is_keyword_hit'].apply(
            lambda x: "✅ 命中" if x else "❌ 未命中"
        )
        details_df['预期关键词'] = details_df['expected_keywords'].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
        display_df = details_df[['query', '预期关键词', '命中排名', '倒数排名', '关键词命中状态']].copy()
        display_df.columns = ['查询问题', '预期关键词', '命中排名', '倒数排名分数', '关键词命中状态']
    elif 'is_chunk_id_hit' in details_df.columns:
        details_df['ID命中状态'] = details_df.apply(
            lambda row: "✅ 命中" if row['is_chunk_id_hit'] else ("❌ 未命中" if row.get('expected_chunk_ids') else "➖ 无预期ID"), 
            axis=1
        )
        display_df = details_df[['query', '命中排名', '倒数排名', 'ID命中状态']].copy()
        display_df.columns = ['查询问题', '命中排名', '倒数排名分数', '预期ID命中状态']
    else:
        display_df = details_df[['query', '命中排名', '倒数排名']].copy()
        display_df.columns = ['查询问题', '命中排名', '倒数排名分数']
        
    st.dataframe(display_df, use_container_width=True, height=300)
else:
    st.warning("未找到测试报告文件")

st.markdown("---")

# ==================== 3. 召回链路透视表 ====================
st.header("🔍 召回链路透视表 (Retrieval Pipeline Table)")
st.markdown("**展示内容：** 展示 hybrid_retrieve 返回的调试数据，用一个交互式表格列出 Top 候选分块")
st.markdown("**表格列名：** Chunk ID、文本片段、BM25原始分、向量原始分、BM25归一化分、向量归一化分、最终融合分")
st.markdown("**作用：** 这是白盒最核心的部分，清晰展示到底是哪个算法把这条数据捞上来的")
st.info("💡 **归一化分**：当前检索结果内的相对分数，1 代表本次召回中相关性最高")

# 真实检索结果
@st.cache_resource
def get_retriever():
    from retrieval.hybrid import HybridRetriever
    from retrieval import bm25 as bm25_mod
    from retrieval.vector_store import load_index as load_faiss_index
    from config.settings import SETTINGS
    import json
    
    # 找到最新的 hash
    hash_db_path = Path("data/hash_db.json")
    if not hash_db_path.exists():
        return None
        
    with open(hash_db_path, 'r', encoding='utf-8') as f:
        hash_db = json.load(f)
        
    if not hash_db:
        return None
        
    # 获取第一个文件的 hash
    file_hash = list(hash_db.keys())[0]
    
    bm25_index = bm25_mod.load_index(file_hash)
    vector_index = load_faiss_index(file_hash, SETTINGS.embedding_model_name)
    
    graph_path = SETTINGS.graph_dir / f"{file_hash}_semantic_graph.json"
    
    return HybridRetriever(
        file_hash=file_hash,
        bm25=bm25_index,
        vector=vector_index,
        graph_path=graph_path,
    )

@st.cache_data(show_spinner=False)
def cached_llm_answer(query: str, context_text: str) -> str:
    """调用 LLM 生成答案，按 query 缓存，避免重复请求"""
    from generation.llm import generate_answer
    from config.settings import SETTINGS
    return generate_answer(
        query=query,
        context_text=context_text,
        model=SETTINGS.llm_model,
        api_key=SETTINGS.llm_api_key or None,
        base_url=SETTINGS.llm_base_url or None,
    )

if test_report and test_queries:
    st.subheader("🔎 选择查询进行检索演示")

    # 选择查询
    query_options = [q['query'] for q in test_queries]
    selected_query = st.selectbox("选择一个查询：", query_options)

    retriever = get_retriever()

    if retriever:
        # 调用真实检索
        results, debug_info = retriever.hybrid_retrieve(selected_query, top_k=5, return_debug_info=True)
        
        if results and debug_info:
            # 提取 debug_info 中 top_k 的数据
            top_chunk_ids = [r['chunk_id'] for r in results]
            top_debug_info = [d for d in debug_info if d['chunk_id'] in top_chunk_ids]
            # 按照 results 的顺序排序 debug_info
            top_debug_info.sort(key=lambda x: top_chunk_ids.index(x['chunk_id']))
            
            # 创建真实数据表格
            real_results = []
            for r, d in zip(results, top_debug_info):
                real_results.append({
                    'Chunk ID': r['chunk_id'],
                    '文本片段': r['text'][:80] + '...' if len(r['text']) > 80 else r['text'],
                    'BM25原始分': round(r['bm25_score'], 4),
                    '向量原始分': round(r['vector_score'], 4),
                    'BM25归一化分': round(d['norm_bm25'], 4),
                    '向量归一化分': round(d['norm_vector'], 4),
                    '最终融合分': round(r['final_score'], 4)
                })
            
            results_df = pd.DataFrame(real_results)
            
            # 使用颜色标记最高分
            def highlight_max(s):
                is_max = s == s.max()
                return ['background-color: #90EE90' if v else '' for v in is_max]
            
            styled_df = results_df.style.apply(highlight_max, subset=['BM25原始分', '向量原始分', 'BM25归一化分', '向量归一化分', '最终融合分'])
            st.dataframe(styled_df, use_container_width=True, height=250)

            # ==================== LLM 生成答案 ====================
            st.markdown("---")
            st.subheader("🤖 LLM 生成答案")
            context_text = "\n\n---\n\n".join([r["text"] for r in results])
            with st.spinner("正在调用 LLM 生成答案…"):
                try:
                    answer = cached_llm_answer(selected_query, context_text)
                    st.info(answer)
                except Exception as e:
                    st.error(f"LLM 调用失败：{e}")

            st.markdown("---")

            # ==================== 4. 分数对比柱状图 ====================
            st.header("📊 分数对比柱状图 (Score Comparison Chart)")
            st.markdown("**展示内容：** 用双柱状图（或堆叠图）展示 Top 5 结果的 norm_bm25 和 norm_vector")
            st.markdown("**作用：** 图形化展示关键词匹配和语义匹配在最终得分里的占比情况")

            # 创建真实的分数对比数据
            comparison_data = {
                'Chunk': [f"Top {i+1} ({d['chunk_id'][:8]}...)" for i, d in enumerate(top_debug_info)],
                'BM25分数': [round(d['norm_bm25'], 4) for d in top_debug_info],
                '向量分数': [round(d['norm_vector'], 4) for d in top_debug_info]
            }
            
            df_comparison = pd.DataFrame(comparison_data)
    else:
        st.warning("无法初始化检索器，请确保索引已构建。")
    
    # 创建分组柱状图
    fig = go.Figure(data=[
        go.Bar(name='BM25分数', x=df_comparison['Chunk'], y=df_comparison['BM25分数'], 
               marker_color='#FF6B6B', text=df_comparison['BM25分数'], textposition='auto'),
        go.Bar(name='向量分数', x=df_comparison['Chunk'], y=df_comparison['向量分数'], 
               marker_color='#4ECDC4', text=df_comparison['向量分数'], textposition='auto')
    ])
    
    fig.update_layout(
        title='Top 5 检索结果的BM25与向量分数对比',
        xaxis_title='排名',
        yaxis_title='归一化分数',
        barmode='group',
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 添加堆叠图选项
    st.subheader("📈 堆叠视图")
    fig_stacked = go.Figure(data=[
        go.Bar(name='BM25分数', x=df_comparison['Chunk'], y=df_comparison['BM25分数'], 
               marker_color='#FF6B6B'),
        go.Bar(name='向量分数', x=df_comparison['Chunk'], y=df_comparison['向量分数'], 
               marker_color='#4ECDC4')
    ])
    
    fig_stacked.update_layout(
        title='分数堆叠视图',
        xaxis_title='排名',
        yaxis_title='累计分数',
        barmode='stack',
        height=400
    )
    
    st.plotly_chart(fig_stacked, use_container_width=True)

st.markdown("---")

# ==================== 页脚信息 ====================
st.markdown("### 💡 系统说明")
st.info("""
**数据来源：**
- 分块数据：`index/*.chunks.jsonl`
- 评估报告：`output/test_report.json`
- 测试查询：`tests/test_queries.json`

**功能特点：**
- ✅ 实时展示文档处理统计
- ✅ 检索质量评估指标
- ✅ 召回链路透明化展示
- ✅ 多维度分数对比可视化
""")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 系统配置")
    st.markdown("**当前数据集：**")
    st.code("your.pdf")
    
    st.markdown("**统计信息：**")
    if chunks_data:
        st.write(f"- 文档块数：{len(chunks_data)}")
    if test_report:
        st.write(f"- 测试查询：{test_report['total_queries']} 条")
        st.write(f"- Recall@5：{test_report['recall_at_5']:.1%}")
        st.write(f"- MRR：{test_report['mrr']:.3f}")
    
    st.markdown("---")
    st.markdown("**🔄 刷新数据**")
    if st.button("重新加载数据"):
        st.cache_data.clear()
        st.rerun()
