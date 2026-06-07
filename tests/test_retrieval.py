"""
测试检索功能
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("初始化 RAG 管道...")
from src.rag_pipeline import RAGPipeline

rag = RAGPipeline()

print("加载知识库...")
try:
    rag.load_knowledge_base()
    print("知识库加载成功，共 %d 个文档块" % len(rag.vector_store))
except Exception as e:
    print("加载失败:", e)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n测试检索...")
question = "什么是 RAG？"
print("问题:", question)

try:
    contexts, sources = rag.retriever.retrieve(question)
    print("检索结果:")
    print("上下文数量:", len(contexts))
    print("来源:", sources)
    if contexts:
        print("第一个上下文:", contexts[0][:100] + "...")
except Exception as e:
    print("检索失败:", e)
    import traceback
    traceback.print_exc()
