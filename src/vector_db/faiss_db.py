"""
FAISS 向量数据库实现

FAISS (Facebook AI Similarity Search) 是 Facebook 开源的高效向量检索库。

优势：
- 性能比内存实现快 100 倍以上
- 支持大规模向量数据（百万级）
- 支持多种索引类型和距离度量

注意：混合检索已移至 HybridRetriever，本类只负责纯向量检索。
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

try:
    import faiss
except ImportError as e:
    raise ImportError(f"FAISS 不可用，请安装 faiss-cpu: {e}")

from .base import VectorDB


class FAISSVectorDB(VectorDB):
    """
    FAISS 向量数据库实现
    
    存储结构：
      - FAISS Index: 用于向量检索
      - chunks: 原始文本块列表
      - sources: 来源文件名列表
    """

    def __init__(self):
        self._index = None
        self.chunks: List[str] = []
        self.sources: List[str] = []

    def _init_index(self, dimension: int):
        """初始化 FAISS 索引"""
        self._index = faiss.IndexFlatIP(dimension)

    def add(self, chunks: List[str], vectors: np.ndarray, sources: Optional[List[str]] = None) -> None:
        """添加文档块及其向量到数据库"""
        if vectors.size == 0:
            return

        # 初始化索引（如果尚未初始化）
        if self._index is None:
            self._init_index(vectors.shape[1])

        # 归一化向量（FAISS IndexFlatIP 要求归一化）
        vectors = vectors.astype(np.float32)
        faiss.normalize_L2(vectors)

        # 添加到 FAISS 索引
        self._index.add(vectors)

        # 添加到文本存储
        self.chunks.extend(chunks)
        if sources:
            self.sources.extend(sources)

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """检索与查询向量最相似的文档块"""
        if self._index is None or len(self.chunks) == 0:
            return []

        # 归一化查询向量
        query_vector = query_vector.astype(np.float32)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        faiss.normalize_L2(query_vector)

        # FAISS 检索
        distances, indices = self._index.search(query_vector, min(top_k, len(self.chunks)))

        results = []
        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            if idx == -1:
                continue
            source = self.sources[idx] if idx < len(self.sources) else "未知来源"
            results.append((self.chunks[idx], float(distances[0][i]), source))

        return results

    def save(self, dir_path: str, doc_metadata: Optional[dict] = None) -> None:
        """将向量数据库持久化到磁盘"""
        save_dir = Path(dir_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self._index is None or len(self.chunks) == 0:
            raise ValueError("向量数据库为空，无法保存")

        # 保存 FAISS 索引
        faiss.write_index(self._index, str(save_dir / "faiss_index.index"))

        # 保存文本块和来源
        with open(save_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        with open(save_dir / "sources.json", "w", encoding="utf-8") as f:
            json.dump(self.sources, f, ensure_ascii=False, indent=2)

        # 保存文档元数据
        if doc_metadata is not None:
            with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(doc_metadata, f, ensure_ascii=False, indent=2)

    def load(self, dir_path: str) -> Optional[dict]:
        """从磁盘加载向量数据库"""
        load_dir = Path(dir_path)
        index_path = load_dir / "faiss_index.index"
        chunks_path = load_dir / "chunks.json"

        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"向量数据库文件不存在: {load_dir}")

        # 加载 FAISS 索引
        self._index = faiss.read_index(str(index_path))

        # 加载文本块和来源
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        sources_path = load_dir / "sources.json"
        if sources_path.exists():
            with open(sources_path, "r", encoding="utf-8") as f:
                self.sources = json.load(f)

        # 加载元数据
        metadata_path = load_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                result = json.load(f)
        else:
            result = None

        return result

    def __len__(self) -> int:
        """返回存储的文档块数量"""
        return len(self.chunks)

    def is_empty(self) -> bool:
        """判断数据库是否为空"""
        return len(self.chunks) == 0
