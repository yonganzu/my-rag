"""
向量数据库抽象接口

定义向量数据库的标准接口，实现多后端支持（如 FAISS、Chroma、Milvus）。

设计原则：
1. 抽象接口与具体实现分离
2. 保持向后兼容性
3. 支持切换不同后端
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
import numpy as np


class VectorDB(ABC):
    """
    向量数据库抽象基类
    
    所有向量数据库实现都应继承此类并实现抽象方法。
    
    注意：混合检索已移至 HybridRetriever，本接口只负责纯向量检索。
    """

    @abstractmethod
    def add(self, chunks: List[str], vectors: np.ndarray, sources: Optional[List[str]] = None) -> None:
        """
        添加文档块及其向量到数据库
        
        参数：
          chunks: 文本块列表
          vectors: 向量矩阵，形状 (n_chunks, dim)
          sources: 来源文件名列表（可选）
        """
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """
        检索与查询向量最相似的文档块
        
        参数：
          query_vector: 查询向量，形状 (1, dim) 或 (dim,)
          top_k: 返回前 k 个结果
        
        返回：
          [(chunk_text, similarity_score, source), ...]，按相似度从高到低排序
        """
        pass

    @abstractmethod
    def save(self, dir_path: str, doc_metadata: Optional[dict] = None) -> None:
        """
        将向量数据库持久化到磁盘
        
        参数：
          dir_path: 保存目录路径
          doc_metadata: 文档元数据（可选，用于检测更新）
        """
        pass

    @abstractmethod
    def load(self, dir_path: str) -> Optional[dict]:
        """
        从磁盘加载向量数据库
        
        参数：
          dir_path: 加载目录路径
        
        返回：
          文档元数据字典（如果存在），否则返回 None
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """返回存储的文档块数量"""
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """判断数据库是否为空"""
        pass


class VectorDBFactory:
    """
    向量数据库工厂类
    
    用于创建不同类型的向量数据库实例。
    
    特性：
      - 支持自动降级：当 FAISS 不可用时自动切换到内存实现
      - 支持显式指定后端类型
    
    使用示例：
        db = VectorDBFactory.create("faiss")  # 优先使用 FAISS
        db = VectorDBFactory.create("memory")  # 使用内存实现
        db = VectorDBFactory.create()          # 自动选择（FAISS 或降级到内存）
    """

    @staticmethod
    def create(db_type: str = "auto") -> VectorDB:
        """
        创建向量数据库实例
        
        参数：
          db_type: 数据库类型，可选值：
            - "faiss": 强制使用 FAISS（如果不可用会抛出异常）
            - "memory": 使用内存实现
            - "auto": 自动选择，优先 FAISS，不可用时降级到内存
        
        返回：
          VectorDB 实例
        """
        if db_type.lower() == "faiss":
            try:
                from .faiss_db import FAISSVectorDB
                print("[向量数据库] 使用 FAISS 后端（高性能）")
                return FAISSVectorDB()
            except ImportError as e:
                print(f"[向量数据库] FAISS 不可用 ({e})，请安装 faiss-cpu")
                raise
        elif db_type.lower() == "memory":
            from .memory_db import MemoryVectorDB
            print("[向量数据库] 使用内存后端（纯 Python，适合小规模数据）")
            return MemoryVectorDB()
        elif db_type.lower() == "auto":
            # 自动选择：优先 FAISS，不可用时降级到内存
            try:
                from .faiss_db import FAISSVectorDB
                print("[向量数据库] 使用 FAISS 后端（高性能）")
                return FAISSVectorDB()
            except ImportError:
                from .memory_db import MemoryVectorDB
                print("[向量数据库] FAISS 不可用，降级到内存后端")
                return MemoryVectorDB()
        else:
            raise ValueError(f"不支持的向量数据库类型: {db_type}")
