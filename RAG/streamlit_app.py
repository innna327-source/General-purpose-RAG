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
    """加载测试查询（优先使用v2标注版本）"""
    from utils.queries import load_test_queries as _load_queries
    # 优先加载v2版本（有has_answer_in_doc标注）
    queries_v2_file = Path("tests/test_queries_v2.json")
    if queries_v2_file.exists():
        with open(queries_v2_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    # 否则使用统一的加载工具
    return _load_queries(Path("tests/test_queries.json"))

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
st.header("🎯 检索评估 vs 答案评估")
st.markdown("**核心概念：** 检索分数高 ≠ 文档有答案")
st.info("""
💡 **两种评估指标的区别：**
- **检索评估**：话题命中率（关键词匹配）—— 检索是否找到了相关话题的文档
- **答案评估**：文档中是否有完整答案 —— 问题能否被检索到的内容回答
""")

if test_report:
    # === 检索评估指标 ===
    st.subheader("📊 检索评估（话题命中率）")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="总查询数",
            value=test_report['total_queries'],
            help="测试集中的查询总数"
        )

    with col2:
        recall_value = test_report.get('retrieval_metrics', {}).get('recall_at_5', test_report.get('recall_at_5', 0))
        st.metric(
            label="话题命中率 (Recall@5)",
            value=f"{recall_value:.2%}",
            delta=f"{recall_value - 0.5:.2%}" if recall_value > 0.5 else None,
            help="前5个结果中命中相关话题的比例（关键词匹配）"
        )

    with col3:
        mrr_value = test_report.get('retrieval_metrics', {}).get('mrr', test_report.get('mrr', 0))
        st.metric(
            label="MRR (平均倒数排名)",
            value=f"{mrr_value:.3f}",
            delta=f"{mrr_value - 0.5:.3f}" if mrr_value > 0.5 else None,
            help="衡量相关结果排名的平均质量"
        )

    # === 答案评估指标 ===
    st.subheader("📝 答案评估（文档是否有答案）")

    answer_metrics = test_report.get('answer_metrics', {})
    queries_with_answer = answer_metrics.get('queries_with_answer', test_report.get('queries_with_answer_in_doc', 0))
    answer_recall = answer_metrics.get('answer_recall')

    col_a1, col_a2, col_a3 = st.columns(3)

    with col_a1:
        st.metric(
            label="文档有答案的问题数",
            value=queries_with_answer,
            help="标注 has_answer_in_doc=true 的问题数量"
        )

    with col_a2:
        queries_without_answer = test_report['total_queries'] - queries_with_answer
        st.metric(
            label="文档无答案的问题数",
            value=queries_without_answer,
            help="标注 has_answer_in_doc=false 的问题数量"
        )

    with col_a3:
        if answer_recall is not None:
            st.metric(
                label="有答案问题召回率",
                value=f"{answer_recall:.2%}",
                delta=f"{answer_recall - 0.5:.2%}" if answer_recall > 0.5 else None,
                help="文档中有答案的问题，检索是否成功命中"
            )
        else:
            st.metric(
                label="有答案问题召回率",
                value="N/A",
                help="需要使用 test_queries_v2.json 标注版本"
            )
        
    # 详细结果表格（区分检索命中和答案存在）
    st.subheader("📋 查询详细结果")
    st.markdown("💡 **注意：** `话题命中 ✅` 但 `文档无答案` = LLM应该拒绝回答")

    details_df = pd.DataFrame(test_report['details'])
    details_df['命中排名'] = details_df['hit_rank'].apply(lambda x: f"第{x}名" if x > 0 else "未命中")
    details_df['倒数排名'] = details_df['reciprocal_rank'].apply(lambda x: f"{x:.3f}")

    # 添加答案状态列
    if 'has_answer_in_doc' in details_df.columns:
        details_df['话题命中'] = details_df['is_keyword_hit'].apply(lambda x: "✅" if x else "❌")
        details_df['文档答案'] = details_df['has_answer_in_doc'].apply(
            lambda x: "✅ 有答案" if x == True else ("❌ 无答案" if x == False else "➖ 未标注")
        )
        details_df['预期关键词'] = details_df['expected_keywords'].apply(lambda x: ", ".join(x[:3]) + "..." if isinstance(x, list) and len(x) > 3 else (", ".join(x) if isinstance(x, list) else str(x)))

        display_df = details_df[['query', '预期关键词', '命中排名', '倒数排名', '话题命中', '文档答案']].copy()
        display_df.columns = ['查询问题', '预期关键词', '命中排名', '倒数排名', '话题命中', '文档答案']

        # 高亮无答案但命中话题的情况（LLM应该拒绝）
        def highlight_no_answer_hit(row):
            if row['话题命中'] == '✅' and row['文档答案'] == '❌ 无答案':
                # 使用深橙色背景 + 黑色文字，确保在黑色主题下可读
                return ['background-color: #8B4513; color: #FFFFFF'] * len(row)
            return [''] * len(row)

        styled_display = display_df.style.apply(highlight_no_answer_hit, axis=1)
        st.dataframe(styled_display, use_container_width=True, height=300)
    elif 'is_keyword_hit' in details_df.columns:
        details_df['关键词命中状态'] = details_df['is_keyword_hit'].apply(
            lambda x: "✅ 命中" if x else "❌ 未命中"
        )
        details_df['预期关键词'] = details_df['expected_keywords'].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
        display_df = details_df[['query', '预期关键词', '命中排名', '倒数排名', '关键词命中状态']].copy()
        display_df.columns = ['查询问题', '预期关键词', '命中排名', '倒数排名分数', '关键词命中状态']
        st.dataframe(display_df, use_container_width=True, height=300)
    elif 'is_chunk_id_hit' in details_df.columns:
        details_df['ID命中状态'] = details_df.apply(
            lambda row: "✅ 命中" if row['is_chunk_id_hit'] else ("❌ 未命中" if row.get('expected_chunk_ids') else "➖ 无预期ID"),
            axis=1
        )
        display_df = details_df[['query', '命中排名', '倒数排名', 'ID命中状态']].copy()
        display_df.columns = ['查询问题', '命中排名', '倒数排名分数', '预期ID命中状态']
        st.dataframe(display_df, use_container_width=True, height=300)
    else:
        display_df = details_df[['query', '命中排名', '倒数排名']].copy()
        display_df.columns = ['查询问题', '命中排名', '倒数排名分数']
        st.dataframe(display_df, use_container_width=True, height=300)
else:
    st.warning("未找到测试报告文件")

st.markdown("---")

# ==================== 3. 召回链路透视表 ====================
st.header("🔍 召回链路透视表 (Retrieval Pipeline Table)")
st.info("💡 **归一化分**：当前检索结果内的相对分数，1 代表本次召回中相关性最高")

# 真实检索结果
@st.cache_resource
def get_retriever():
    from retrieval.factory import load_retriever
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
    return load_retriever(file_hash)

@st.cache_data(show_spinner=False)
def cached_llm_answer(query: str, context_text: str, use_parent: bool) -> str:
    """调用 LLM 生成答案，按 query + context + mode 缓存，避免重复请求"""
    from generation.llm import generate_answer
    from config.settings import SETTINGS
    return generate_answer(
        query=query,
        context_text=context_text,
        model=SETTINGS.generation_llm_model,
        api_key=SETTINGS.llm_api_key or None,
        base_url=SETTINGS.llm_base_url or None,
    )

if test_report and test_queries:
    st.subheader("🔎 选择查询进行检索演示")

    # 父文档召回开关
    use_parent = st.checkbox(
        "🔄 窗口扩展召回（推荐）",
        value=True,
        help="检索用短chunk精确匹配，给LLM时扩展前后各2个相邻chunk，避免信息切断又不过多冗余"
    )

    # 选择查询
    query_options = [q['query'] for q in test_queries]
    selected_query = st.selectbox("选择一个查询：", query_options)

    retriever = get_retriever()

    if retriever:
        # 调用真实检索（根据开关决定是否使用父文档）
        results, debug_info = retriever.hybrid_retrieve(
            selected_query, top_k=5, return_debug_info=True, use_parent_context=use_parent
        )
        
        if results and debug_info:
            # 提取 debug_info 中 top_k 的数据
            top_chunk_ids = [r['chunk_id'] for r in results]
            top_debug_info = [d for d in debug_info if d['chunk_id'] in top_chunk_ids]
            # 按照 results 的顺序排序 debug_info
            top_debug_info.sort(key=lambda x: top_chunk_ids.index(x['chunk_id']))
            fusion_rank_map = {
                d['chunk_id']: i + 1
                for i, d in enumerate(sorted(debug_info, key=lambda x: x.get('final_score', 0), reverse=True))
            }
            rerank_active = any(r.get('rerank_used', False) for r in results)
            rank_changes = [
                fusion_rank_map.get(r['chunk_id'], i + 1) - (i + 1)
                for i, r in enumerate(results)
            ]

            # 显示召回模式说明
            if use_parent:
                st.info(f"💡 **窗口扩展模式**：每个结果包含命中chunk + 前后各2个相邻chunk，信息更完整")
            else:
                st.info(f"💡 **普通模式**：只返回单个短chunk（~200字），可能信息不完整")

            # 创建真实数据表格
            if rerank_active:
                moved_up = sum(1 for delta in rank_changes if delta > 0)
                best_gain = max(rank_changes) if rank_changes else 0
                st.success(f"重排已启用：Top {len(results)} 中 {moved_up} 条结果被提升名次，最大提升 {best_gain} 位。")
            else:
                st.warning("重排未启用或已降级：当前结果按 BM25 + 向量 + 图谱融合分排序。")

            real_results = []
            for idx, (r, d) in enumerate(zip(results, top_debug_info), start=1):
                # 显示文本长度信息
                text_len = len(r['text'])
                chunk_len = len(r.get('chunk_text', r['text'])) if use_parent else text_len

                display_text = r['text'][:60] + '...' if len(r['text']) > 60 else r['text']
                fusion_rank = fusion_rank_map.get(r['chunk_id'], idx)
                rank_delta = fusion_rank - idx

                real_results.append({
                    'Rerank Used': r.get('rerank_used', False),
                    'Rerank Score': round(r.get('rerank_score', 0.0), 4),
                    'Chunk ID': r['chunk_id'],
                    '文本片段': display_text,
                    '文本长度': f"{text_len}字" + (f" (原chunk:{chunk_len}字)" if use_parent and text_len != chunk_len else ""),
                    'BM25原始分': round(r['bm25_score'], 4),
                    '向量原始分': round(r['vector_score'], 4),
                    'BM25归一化分': round(d['norm_bm25'], 4),
                    '向量归一化分': round(d['norm_vector'], 4),
                    '最终排序分（含图谱加权）': round(r['final_score'], 4)
                })
            
            results_df = pd.DataFrame(real_results)
            
            # 使用颜色标记最高分
            def highlight_max(s):
                is_max = s == s.max()
                return ['background-color: #90EE90' if v else '' for v in is_max]
            
            styled_df = results_df.style.apply(highlight_max, subset=['BM25原始分', '向量原始分', 'BM25归一化分', '向量归一化分', '最终排序分（含图谱加权）'])
            st.dataframe(styled_df, use_container_width=True, height=250)

            # ==================== LLM 生成答案 + 答案验证 ====================
            st.markdown("---")
            st.subheader("🤖 LLM 生成答案 + 答案验证")

            # 获取当前查询的标注信息
            current_query_info = next((q for q in test_queries if q['query'] == selected_query), None)
            has_answer_in_doc = current_query_info.get('has_answer_in_doc', None) if current_query_info else None
            answer_key_points = current_query_info.get('answer_key_points', []) if current_query_info else []
            notes = current_query_info.get('notes', '') if current_query_info else ''

            # 显示文档答案状态
            st.markdown("**📝 文档答案状态：**")
            if has_answer_in_doc is True:
                st.success(f"✅ 文档中有答案 | 关键点: {', '.join(answer_key_points[:3]) if answer_key_points else '无'}")
            elif has_answer_in_doc is False:
                st.warning(f"❌ 文档中无答案 | 原因: {notes if notes else '内容缺失'}")
            else:
                st.info("➖ 未标注答案状态")

            # 检测LLM是否返回"无法回答"
            from generation.constants import NO_ANSWER_PHRASES

            # 显示给LLM的上下文信息
            context_text = "\n\n---\n\n".join([r["text"] for r in results])
            total_context_len = len(context_text)

            st.markdown("**📖 LLM接收的上下文：**")
            if use_parent:
                avg_expanded_len = total_context_len // len(results) if results else 0
                st.success(f"✅ 窗口扩展模式：{len(results)} 个扩展片段，共 {total_context_len} 字（平均 {avg_expanded_len} 字/片段）")
            else:
                st.warning(f"⚠️ 普通模式：{len(results)} 个短chunk，共 {total_context_len} 字（可能信息不完整）")

            with st.spinner("正在调用 LLM 生成答案…"):
                try:
                    answer = cached_llm_answer(selected_query, context_text, use_parent)

                    # 判断LLM是否正确处理
                    is_no_answer_response = any(phrase in answer for phrase in NO_ANSWER_PHRASES)

                    st.markdown("**🤖 LLM 回答：**")
                    st.info(answer)

                    # 答案验证
                    st.markdown("**✅ 答案验证：**")
                    if has_answer_in_doc is True:
                        if is_no_answer_response:
                            if use_parent:
                                st.error("❌ **LLM行为错误** - 已使用窗口扩展（前后各2个chunk），但LLM仍拒绝回答")
                            else:
                                st.warning("⚠️ **可能是chunk模式导致** - 请尝试开启窗口扩展以获取完整信息")
                        else:
                            st.success("✅ **LLM行为正确** - 文档有答案，LLM成功回答")
                    elif has_answer_in_doc is False:
                        if is_no_answer_response:
                            st.success("✅ **LLM行为正确** - 文档无答案，LLM正确拒绝回答")
                        else:
                            st.error("❌ **LLM行为错误** - 文档无答案但LLM尝试回答（可能编造内容）")
                    else:
                        st.info("➖ **未标注答案状态**，无法验证LLM行为")

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

    st.markdown("**📊 检索评估：**")
    if test_report:
        recall = test_report.get('retrieval_metrics', {}).get('recall_at_5', test_report.get('recall_at_5', 0))
        mrr = test_report.get('retrieval_metrics', {}).get('mrr', test_report.get('mrr', 0))
        st.write(f"- 话题命中率：{recall:.1%}")
        st.write(f"- MRR：{mrr:.3f}")

    st.markdown("**📝 答案评估：**")
    if test_report:
        queries_with_answer = test_report.get('queries_with_answer_in_doc', 0)
        total = test_report.get('total_queries', 0)
        answer_recall = test_report.get('answer_metrics', {}).get('answer_recall')
        st.write(f"- 有答案问题：{queries_with_answer}/{total}")
        if answer_recall is not None:
            st.write(f"- 有答案召回率：{answer_recall:.1%}")

    st.markdown("---")
    st.markdown("**🔄 刷新数据**")
    if st.button("重新加载数据"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("**💡 评估说明**")
    st.info("""
- **检索分数高** = 找到相关话题
- **文档有答案** = 能回答问题
- 两者独立评估，分开展示
""")
