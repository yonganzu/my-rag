"""
向量数据库模块

提供多种向量数据库实现和混合检索能力。

主要组件：
  - VectorDB: 抽象基类，定义向量数据库接口
  - FAISSVectorDB: FAISS 实现（高性能）
  - MemoryVectorDB: 内存实现（降级方案）
  - BM25Retriever: 独立的 BM25 关键词检索器
  - HybridRetriever: 混合检索器（向量 + BM25）
  - VectorDBFactory: 工厂类，创建不同后端的向量数据库
"""

from .base import VectorDB, VectorDBFactory
from .faiss_db import FAISSVectorDB
from .memory_db import MemoryVectorDB
from .bm25_retriever import BM25Retriever
from .hybrid_retriever import HybridRetriever

__all__ = [
    "VectorDB",
    "VectorDBFactory",
    "FAISSVectorDB",
    "MemoryVectorDB",
    "BM25Retriever",
    "HybridRetriever",
]
