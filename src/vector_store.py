"""
向量存储模块

为什么自己实现而不是用 FAISS / Chroma？
  - 用 200 行代码理解向量检索的核心原理
  - 生产环境再换 FAISS（性能快 100 倍）、Milvus（分布式）或 Chroma（全功能）
  - 核心算法都一样：向量相似度计算，只是工程优化程度不同

核心概念：
  - 余弦相似度（Cosine Similarity）：衡量两个向量方向的接近程度
  - 范围 [-1, 1]，越接近 1 表示越相似
  - 比欧氏距离更适合文本语义匹配（关注方向而非幅度）
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np


def cosine_similarity(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """
    计算查询向量与候选向量集合的余弦相似度

    数学原理：
      cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)

    代码实现：
      query @ candidates.T  = 矩阵乘法，一次算出 query 与所有 candidate 的点积
      norm(query) * norm(candidates) = 归一化因子
    """
    # ── L2 归一化 ────────────────────────────────────────────────
    # 对向量除以其模长，使得所有向量长度为 1
    # 归一化后，点积 = 余弦相似度（简化计算）
    query_norm = query / np.linalg.norm(query, axis=1, keepdims=True)
    cand_norm = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)

    # ── 点积计算相似度 ──────────────────────────────────────────
    # query_norm 形状: (1, dim)
    # cand_norm.T 形状: (dim, n_candidates)
    # 结果形状: (1, n_candidates)
    return query_norm @ cand_norm.T


class VectorStore:
    """
    简单的内存向量存储 + 检索

    存储结构：
      - self.chunks: List[str]         # 原始文本，按索引对应
      - self.vectors: np.ndarray       # 向量矩阵，形状 (n_chunks, dim)
    """

    def __init__(self):
        self.chunks: List[str] = []
        self.vectors: np.ndarray | None = None

    def add(self, chunks: List[str], vectors: np.ndarray) -> None:
        """添加文档块及其向量到存储中"""
        if self.vectors is None:
            self.vectors = vectors
        else:
            # 垂直堆叠（在行方向拼接），增加新的向量行
            self.vectors = np.vstack([self.vectors, vectors])
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 3, fetch_k: int = None) -> List[Tuple[str, float]]:
        """
        检索与查询向量最相似的文档块

        参数：
          query_vector: 查询向量，形状 (1, dim)
          top_k: 返回前 k 个结果
          fetch_k: 先获取 fetch_k 个候选（用于 rerank），默认为 top_k

        返回：
          [(chunk_text, score), ...]，按相似度从高到低排序
        """
        if self.vectors is None or len(self.chunks) == 0:
            return []

        if fetch_k is None:
            fetch_k = top_k

        # ── 计算相似度 ──────────────────────────────────────────
        query_vector = query_vector.reshape(1, -1)  # 确保形状正确
        scores = cosine_similarity(query_vector, self.vectors)

        # ── 取 top-k ────────────────────────────────────────────
        # squeeze() 去掉单维度，argsort() 返回排序索引
        # [::-1] 反转变成从大到小
        fetch_k = min(fetch_k, len(self.chunks))
        top_indices = scores.squeeze().argsort()[::-1][:fetch_k]

        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(scores.squeeze()[idx])))

        return results[:top_k]

    def __len__(self) -> int:
        """返回存储的文档块数量"""
        return len(self.chunks)

    def save(self, dir_path: str | Path, doc_metadata: dict = None) -> None:
        """
        将向量数据库持久化到磁盘

        保存位置：
          - dir_path / vectors.npy       # numpy 格式的向量矩阵
          - dir_path / chunks.json       # JSON 格式的文本块
          - dir_path / metadata.json     # 文档元数据（用于检测更新）
        """
        import json

        save_dir = Path(dir_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.vectors is None or len(self.chunks) == 0:
            raise ValueError("向量存储为空，无法保存")

        np.save(save_dir / "vectors.npy", self.vectors)
        with open(save_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        
        # 保存文档元数据
        if doc_metadata is not None:
            with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(doc_metadata, f, ensure_ascii=False, indent=2)

    def load(self, dir_path: str | Path) -> dict | None:
        """
        从磁盘加载向量数据库

        加载位置：
          - dir_path / vectors.npy       # 向量矩阵
          - dir_path / chunks.json       # 文本块
          - dir_path / metadata.json     # 文档元数据（如果存在）

        返回：
          文档元数据字典（如果存在），否则返回 None
        """
        import json

        load_dir = Path(dir_path)
        vectors_path = load_dir / "vectors.npy"
        chunks_path = load_dir / "chunks.json"
        metadata_path = load_dir / "metadata.json"

        if not vectors_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"向量数据库文件不存在: {load_dir}")

        self.vectors = np.load(vectors_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        
        # 加载元数据（如果存在）
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
