import json
import re

SYNONYM_DICT = {
    "rag": ["检索增强", "检索", "rag", "retrieval augmented generation"],
    "微调": ["fine-tuning", "sft", "微调", "finetune", "lora", "全量微调", "指令微调", "prompt tuning"],
    "向量库": ["向量数据库", "milvus", "weaviate", "chroma", "qdrant", "向量", "vector database"],
    "agent": ["智能体", "agent", "多智能体", "工作流", "单智能体", "multi-agent"],
    "图谱": ["graphrag", "知识图谱", "图", "实体", "关系", "全局检索", "局部检索"],
    "切割": ["分块", "切块", "切割", "chunk", "滑动窗口", "重叠", "文档切割", "递归字符"],
    "召回": ["召回", "检索", "recall", "重排序", "rerank", "混合检索", "向量检索", "关键词检索", "bm25"],
    "评估": ["评估", "评测", "ragas", "忠诚度", "相关性", "mrr", "recall", "指标"],
    "记忆": ["记忆", "上下文", "memory", "短期记忆", "长期记忆", "kv cache"],
    "调用": ["调用", "function calling", "工具调用", "mcp", "skill", "意图识别"],
    "幻觉": ["幻觉", "hallucination", "瞎编", "胡说八道"],
    "推理": ["推理", "加速", "vllm", "量化", "剪枝", "蒸馏", "模型压缩"],
    "改写": ["改写", "query改写", "hyde", "拆解", "decomposition", "step-back"],
    "大模型": ["大模型", "llm", "模型", "deepseek", "openai"]
}

def get_synonyms(word):
    word_lower = word.lower()
    for key, synonyms in SYNONYM_DICT.items():
        if word_lower in key or word_lower in synonyms:
            return synonyms
    
    # Try partial matching
    for key, synonyms in SYNONYM_DICT.items():
        for syn in synonyms:
            if word_lower in syn or syn in word_lower:
                return synonyms
                
    return []

def extract_keywords(query):
    # Remove common stop words and punctuation to get keywords
    stopwords = ["的", "是", "了", "吗", "呢", "啊", "怎么", "什么", "到底", "哪个", "好", "可以", "能", "有哪些", "为什么", "为啥", "如何", "是不是", "和", "与", "在", "里", "用", "做", "对", "把", "去", "有", "没"]
    
    # Simple word segmentation using regex for non-Chinese characters and keeping Chinese characters grouped
    words = re.findall(r'[a-zA-Z0-9]+|[\u4e00-\u9fa5]{2,}', query)
    
    # Filter out stopwords
    keywords = []
    for w in words:
        if w not in stopwords:
            keywords.append(w)
            
    # If no keywords found, fallback
    if not keywords:
        for w in re.findall(r'[a-zA-Z0-9]+|[\u4e00-\u9fa5]', query):
            if w not in stopwords:
                keywords.append(w)
                
    # Expand with synonyms to create a rich expected_keywords list
    expanded_keywords = set()
    for k in keywords:
        expanded_keywords.add(k)
        syns = get_synonyms(k)
        for s in syns:
            expanded_keywords.add(s)
            
    # For some domain specific queries, even if word segmentation missed it, we can check if it exists in the query
    for key, synonyms in SYNONYM_DICT.items():
        for syn in synonyms:
            if syn in query.lower():
                expanded_keywords.update(synonyms)
                break
                
    result = list(expanded_keywords)
    
    # If still empty, use query substrings
    if not result:
        result = [query[:2]] if len(query) >= 2 else [query]
            
    return result

def main():
    input_path = "tests/test_queries.json"
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if not data:
        print("Empty file")
        return
        
    # We might be reading a list of strings OR a list of dicts (since we already converted once)
    new_data = []
    for item in data:
        if isinstance(item, str):
            query = item
        elif isinstance(item, dict) and "query" in item:
            query = item["query"]
        else:
            continue
            
        keywords = extract_keywords(query)
            
        new_data.append({
            "query": query,
            "expected_keywords": keywords
        })
        
    with open(input_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully enriched {len(new_data)} queries with synonyms in expected_keywords.")

if __name__ == "__main__":
    main()
