"""
内存向量数据库实现（降级方案）

当 FAISS 不可用时使用此实现，保持向后兼容。

特点：
- 纯 Python 实现，无需额外依赖
- 适合小规模数据集（<10000 个文档块）
- 性能比 FAISS 慢约 100 倍

注意：混合检索已移至 HybridRetriever，本类只负责纯向量检索。
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

from .base import VectorDB


def cosine_similarity(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """计算查询向量与候选向量集合的余弦相似度"""
    query_norm = query / np.linalg.norm(query, axis=1, keepdims=True)
    cand_norm = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)
    return query_norm @ cand_norm.T


class MemoryVectorDB(VectorDB):
    """
    内存向量数据库实现（降级方案）
    
    存储结构：
      - self.chunks: 原始文本块列表
      - self.vectors: 向量矩阵（numpy array）
      - self.sources: 来源文件名列表
    """

    def __init__(self):
        self.chunks: List[str] = []
        self.sources: List[str] = []
        self.vectors: Optional[np.ndarray] = None

    def add(self, chunks: List[str], vectors: np.ndarray, sources: Optional[List[str]] = None) -> None:
        """添加文档块及其向量到数据库"""
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        self.chunks.extend(chunks)
        if sources:
            self.sources.extend(sources)

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """检索与查询向量最相似的文档块"""
        if self.vectors is None or len(self.chunks) == 0:
            return []

        query_vector = query_vector.reshape(1, -1)
        scores = cosine_similarity(query_vector, self.vectors)

        # scores 形状: (1, n)。使用 flatten 防止 squeeze 在 n=1 时产生 0 维数组
        scores_flat = scores.flatten()
        top_indices = scores_flat.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            idx = int(idx)
            source = self.sources[idx] if idx < len(self.sources) else "未知来源"
            results.append((self.chunks[idx], float(scores_flat[idx]), source))

        return results

    def save(self, dir_path: str, doc_metadata: Optional[dict] = None) -> None:
        """将向量数据库持久化到磁盘"""
        save_dir = Path(dir_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.vectors is None or len(self.chunks) == 0:
            raise ValueError("向量数据库为空，无法保存")

        np.save(save_dir / "vectors.npy", self.vectors)
        with open(save_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        with open(save_dir / "sources.json", "w", encoding="utf-8") as f:
            json.dump(self.sources, f, ensure_ascii=False, indent=2)

        if doc_metadata is not None:
            with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(doc_metadata, f, ensure_ascii=False, indent=2)

    def load(self, dir_path: str) -> Optional[dict]:
        """从磁盘加载向量数据库"""
        load_dir = Path(dir_path)
        vectors_path = load_dir / "vectors.npy"
        chunks_path = load_dir / "chunks.json"

        if not vectors_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"向量数据库文件不存在: {load_dir}")

        self.vectors = np.load(vectors_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        sources_path = load_dir / "sources.json"
        if sources_path.exists():
            with open(sources_path, "r", encoding="utf-8") as f:
                self.sources = json.load(f)

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
