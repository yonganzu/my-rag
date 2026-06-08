"""
内存向量数据库实现（降级方案）

当 FAISS 不可用时使用此实现，保持向后兼容。

特点：
- 纯 Python 实现，无需额外依赖
- 适合小规模数据集（<10000 个文档块）
- 性能比 FAISS 慢约 100 倍
"""

import json
import math
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np

from .base import VectorDB


def cosine_similarity(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """计算查询向量与候选向量集合的余弦相似度"""
    query_norm = query / np.linalg.norm(query, axis=1, keepdims=True)
    cand_norm = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)
    return query_norm @ cand_norm.T


def _tokenize(text: str) -> List[str]:
    """简单分词：中英文混合处理"""
    try:
        import jieba
    except ImportError:
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+', text.lower())
        return [w for w in words if len(w) > 1]

    chinese_parts = re.findall(r'[\u4e00-\u9fff]+', text)
    for part in chinese_parts:
        words = list(jieba.cut(part))
        text = text.replace(part, ' '.join(words))
    tokens = re.findall(r'[a-zA-Z0-9]+|[\u4e00-\u9fff]+', text.lower())
    return [t for t in tokens if len(t) > 1]


class BM25Index:
    """BM25 关键词索引"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0
        self.idf: Dict[str, float] = {}
        self._doc_count: int = 0
        self._term_doc_freq: Dict[str, int] = {}

    def build(self, chunks: List[str]) -> None:
        """构建 BM25 索引"""
        self.documents = []
        self.doc_lengths = []
        self._term_doc_freq = {}

        for chunk in chunks:
            tokens = _tokenize(chunk)
            self.documents.append(tokens)
            self.doc_lengths.append(len(tokens))
            unique_terms = set(tokens)
            for term in unique_terms:
                self._term_doc_freq[term] = self._term_doc_freq.get(term, 0) + 1

        self._doc_count = len(self.documents)
        if self._doc_count == 0:
            return

        self.avgdl = sum(self.doc_lengths) / self._doc_count

        for term, df in self._term_doc_freq.items():
            self.idf[term] = math.log(
                (self._doc_count - df + 0.5) / (df + 0.5) + 1.0
            )

    def add_documents(self, chunks: List[str]) -> List[int]:
        """增量添加文档到索引"""
        start_idx = len(self.documents)
        for chunk in chunks:
            tokens = _tokenize(chunk)
            self.documents.append(tokens)
            self.doc_lengths.append(len(tokens))
            unique_terms = set(tokens)
            for term in unique_terms:
                self._term_doc_freq[term] = self._term_doc_freq.get(term, 0) + 1

        self._doc_count = len(self.documents)
        self.avgdl = sum(self.doc_lengths) / self._doc_count

        for term, df in self._term_doc_freq.items():
            self.idf[term] = math.log(
                (self._doc_count - df + 0.5) / (df + 0.5) + 1.0
            )

        return list(range(start_idx, len(self.documents)))

    def search(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """BM25 检索"""
        if self._doc_count == 0:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = np.zeros(self._doc_count)
        for token in set(query_tokens):
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for i, doc_tokens in enumerate(self.documents):
                tf = doc_tokens.count(token)
                if tf == 0:
                    continue
                doc_len = self.doc_lengths[i]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                scores[i] += idf_val * numerator / denominator

        top_indices = scores.argsort()[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]

    def save(self, dir_path: str) -> None:
        """持久化 BM25 索引"""
        save_dir = Path(dir_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "k1": self.k1,
            "b": self.b,
            "documents": self.documents,
            "idf": self.idf,
            "doc_lengths": self.doc_lengths,
            "avgdl": self.avgdl,
            "term_doc_freq": self._term_doc_freq,
            "doc_count": self._doc_count,
        }
        with open(save_dir / "bm25_index.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, dir_path: str) -> bool:
        """从磁盘加载 BM25 索引"""
        load_dir = Path(dir_path)
        path = load_dir / "bm25_index.json"
        if not path.exists():
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.k1 = data["k1"]
        self.b = data["b"]
        self.documents = data["documents"]
        self.idf = data["idf"]
        self.doc_lengths = data["doc_lengths"]
        self.avgdl = data["avgdl"]
        self._term_doc_freq = data["term_doc_freq"]
        self._doc_count = data["doc_count"]
        return True


class MemoryVectorDB(VectorDB):
    """
    内存向量数据库实现（降级方案）
    
    存储结构：
      - self.chunks: 原始文本块列表
      - self.vectors: 向量矩阵（numpy array）
      - self.sources: 来源文件名列表
      - self.bm25: BM25Index 用于关键词检索
    """

    def __init__(self):
        self.chunks: List[str] = []
        self.sources: List[str] = []
        self.vectors: Optional[np.ndarray] = None
        self.bm25 = BM25Index()

    def add(self, chunks: List[str], vectors: np.ndarray, sources: Optional[List[str]] = None) -> None:
        """添加文档块及其向量到数据库"""
        if self.vectors is None:
            self.vectors = vectors
            self.bm25.build(chunks)
        else:
            self.vectors = np.vstack([self.vectors, vectors])
            self.bm25.add_documents(chunks)
        self.chunks.extend(chunks)
        if sources:
            self.sources.extend(sources)

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """检索与查询向量最相似的文档块"""
        if self.vectors is None or len(self.chunks) == 0:
            return []

        query_vector = query_vector.reshape(1, -1)
        scores = cosine_similarity(query_vector, self.vectors)

        top_indices = scores.squeeze().argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            idx = int(idx)
            source = self.sources[idx] if idx < len(self.sources) else "未知来源"
            results.append((self.chunks[idx], float(scores.squeeze()[idx]), source))

        return results

    def hybrid_search(
        self,
        query_text: str,
        query_vector: np.ndarray,
        top_k: int = 3,
        bm25_weight: float = 0.5,
    ) -> List[Tuple[str, float, str]]:
        """混合检索：向量语义 + BM25 关键词，用 RRF 融合"""
        if self.vectors is None or len(self.chunks) == 0:
            return []

        RRF_K = 60
        fetch_k = top_k * 3

        # -- 1. 向量检索 --
        qv = query_vector.reshape(1, -1)
        vec_scores = cosine_similarity(qv, self.vectors).squeeze()
        vec_ranked = vec_scores.argsort()[::-1][:fetch_k]

        # -- 2. BM25 检索 --
        bm25_results = self.bm25.search(query_text, top_k=fetch_k)

        # -- 3. RRF 融合 --
        rrf_scores: Dict[int, float] = {}
        for rank, idx in enumerate(vec_ranked):
            idx = int(idx)
            rrf_scores[idx] = rrf_scores.get(idx, 0) + (1.0 - bm25_weight) / (RRF_K + rank + 1)

        for rank, (idx, bm25_score) in enumerate(bm25_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + bm25_weight / (RRF_K + rank + 1)

        # -- 4. 按 RRF 分数排序 --
        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:top_k]
        results = []
        for idx in sorted_indices:
            source = self.sources[idx] if idx < len(self.sources) else "未知来源"
            results.append((self.chunks[idx], rrf_scores[idx], source))

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

        self.bm25.save(save_dir)

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

        self.bm25.load(load_dir)
        return result

    def __len__(self) -> int:
        """返回存储的文档块数量"""
        return len(self.chunks)

    def is_empty(self) -> bool:
        """判断数据库是否为空"""
        return len(self.chunks) == 0
