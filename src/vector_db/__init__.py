"""
向量数据库模块

提供多种向量数据库后端支持：
- FAISS（推荐，高性能）
- Memory（降级方案，纯 Python）

使用方式：
    from src.vector_db import VectorDBFactory
    
    # 创建向量数据库（自动选择后端）
    db = VectorDBFactory.create()
    
    # 强制使用内存后端
    db = VectorDBFactory.create("memory")
"""

from .base import VectorDB, VectorDBFactory

# 不直接导入 FAISSVectorDB，避免启动时就需要 FAISS
# FAISS 将在工厂类中动态导入

__all__ = ["VectorDB", "VectorDBFactory"]
