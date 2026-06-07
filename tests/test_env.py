"""
测试环境和基本功能
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("测试环境变量...")
api_key = os.environ.get("DASHSCOPE_API_KEY")
if api_key:
    print("DASHSCOPE_API_KEY: 设置成功 (长度: %d)" % len(api_key))
else:
    print("DASHSCOPE_API_KEY: 未设置")

print("\n测试导入模块...")
try:
    from src.config import config
    print("config 模块导入成功")
    print("LLM 模型:", config.llm_model)
    print("Embedding 模型:", config.embedding_model)
except Exception as e:
    print("导入 config 失败:", e)

print("\n测试向量数据库...")
try:
    from src.vector_store import VectorStore
    vs = VectorStore()
    print("VectorStore 创建成功")
    
    db_path = project_root / "data" / "vector_db"
    if db_path.exists():
        print("向量数据库路径存在")
        files = list(db_path.iterdir())
        print("文件列表:", [f.name for f in files])
    else:
        print("向量数据库路径不存在")
except Exception as e:
    print("测试 VectorStore 失败:", e)
