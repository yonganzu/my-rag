"""
测试重排序方法切换功能
"""
from src.rag_pipeline import RAGPipeline
from src.config import config

print('测试重排序方法切换功能')
print('=' * 60)

# 测试向量相似度重排序（免费）
print('\n1. 测试向量相似度重排序（免费）')
config.rerank_method = 'vector'
rag = RAGPipeline()
rag.load_knowledge_base()
answer, contexts = rag.answer('LLM是什么?', use_rerank=True, use_query_rewrite=False, show_citations=False)
print('答案:', answer[:100], '...')
