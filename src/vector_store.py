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

BM25 索引：
  - 经典的基于关键词的检索算法
  - 与向量检索互补：BM25 擅长精确匹配，向量擅长语义泛化
  - 混合检索融合两者优势，取长补短
"""

import json
import math
import re
from pathlib import Path
from typing import List, Tuple, Dict

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


def _tokenize(text: str) -> List[str]:
    """
    简单分词：中英文混合处理

    - 中文：使用 jieba 分词
    - 英文/数字：按空格和标点切分
    """

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
    """
    BM25 关键词索引

    公式：
      score(D, Q) = Σ IDF(qi) × (TF(qi,D) × (k1+1)) / (TF(qi,D) + k1×(1-b+b×|D|/avgdl))

    k1: 词频饱和度（防止词频过高主导评分）
    b:  文档长度归一化（长文档不天然占优）
    """

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
        """增量添加文档到索引，返回新文档的索引列表"""
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
        """
        BM25 检索

        返回：
          [(chunk_index, bm25_score), ...]，按分数从高到低排序
        """
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

    def save(self, dir_path: str | Path) -> None:
        """持久化 BM25 索引到磁盘"""
        save_dir = Path(dir_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "k1": self.k1,
            "b": self.b,
            "documents": [["\x01".join(tokens) for tokens in self.documents]],
            "idf": self.idf,
            "doc_lengths": self.doc_lengths,
            "avgdl": self.avgdl,
            "term_doc_freq": self._term_doc_freq,
            "doc_count": self._doc_count,
        }
        with open(save_dir / "bm25_index.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, dir_path: str | Path) -> bool:
        """从磁盘加载 BM25 索引"""
        import json as _json
        load_dir = Path(dir_path)
        path = load_dir / "bm25_index.json"
        if not path.exists():
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        self.k1 = data["k1"]
        self.b = data["b"]
        raw_docs = data.get("documents", [])
        if raw_docs and isinstance(raw_docs[0], list) and len(raw_docs) == 1:
            raw_docs = raw_docs[0]
        self.documents = [[t for t in tokens if t != "\x01"] if isinstance(tokens, list) else tokens.split("\x01") for tokens in raw_docs]
        self.idf = data["idf"]
        self.doc_lengths = data["doc_lengths"]
        self.avgdl = data["avgdl"]
        self._term_doc_freq = data["term_doc_freq"]
        self._doc_count = data["doc_count"]
        return True


class VectorStore:
    """
    简单的内存向量存储 + 检索

    存储结构：
      - self.chunks: List[str]         # 原始文本，按索引对应
      - self.vectors: np.ndarray       # 向量矩阵，形状 (n_chunks, dim)
    """

    def __init__(self):
        self.chunks: List[str] = []
        self.sources: List[str] = []  # 每个 chunk 的来源文件名
        self.vectors: np.ndarray | None = None
        self.bm25 = BM25Index()

    def add(self, chunks: List[str], vectors: np.ndarray, sources: List[str] = None) -> None:
        """添加文档块及其向量到存储中"""
        if self.vectors is None:
            self.vectors = vectors
            self.bm25.build(chunks)
        else:
            self.vectors = np.vstack([self.vectors, vectors])
            self.bm25.add_documents(chunks)
        self.chunks.extend(chunks)
        if sources:
            self.sources.extend(sources)

    def search(self, query_vector: np.ndarray, top_k: int = 3, fetch_k: int = None) -> List[Tuple[str, float, str]]:
        """
        检索与查询向量最相似的文档块

        参数：
          query_vector: 查询向量，形状 (1, dim)
          top_k: 返回前 k 个结果
          fetch_k: 先获取 fetch_k 个候选（用于 rerank），默认为 top_k

        返回：
          [(chunk_text, score, source), ...]，按相似度从高到低排序
        """
        if self.vectors is None or len(self.chunks) == 0:
            return []

        if fetch_k is None:
            fetch_k = top_k

        query_vector = query_vector.reshape(1, -1)
        scores = cosine_similarity(query_vector, self.vectors)

        fetch_k = min(fetch_k, len(self.chunks))
        top_indices = scores.squeeze().argsort()[::-1][:fetch_k]

        results = []
        for idx in top_indices:
            source = self.sources[idx] if idx < len(self.sources) else "未知来源"
            results.append((self.chunks[idx], float(scores.squeeze()[idx]), source))

        return results[:top_k]

    def hybrid_search(
        self,
        query_text: str,
        query_vector: np.ndarray,
        top_k: int = 3,
        fetch_k: int = None,
        bm25_weight: float = 0.5,
    ) -> List[Tuple[str, float, str]]:
        """
        混合检索：向量语义 + BM25 关键词，用 RRF 融合

        RRF（Reciprocal Rank Fusion）公式：
          RRF_score(d) = Σ (1 / (k + rank_i(d)))

        其中 k=60，rank_i(d) 是文档 d 在第 i 个检索器中的排名

        返回：
          [(chunk_text, rrf_score, source), ...]，按 RRF 分数从高到低排序
        """
        if self.vectors is None or len(self.chunks) == 0:
            return []

        if fetch_k is None:
            fetch_k = top_k

        RRF_K = 60

        # -- 1. 向量检索 --
        qv = query_vector.reshape(1, -1)
        vec_scores = cosine_similarity(qv, self.vectors).squeeze()
        vec_ranked = vec_scores.argsort()[::-1][:fetch_k]

        # -- 2. BM25 检索 --
        bm25_results = self.bm25.search(query_text, top_k=fetch_k)
        bm25_indexed = {idx: score for idx, score in bm25_results}

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
        with open(save_dir / "sources.json", "w", encoding="utf-8") as f:
            json.dump(self.sources, f, ensure_ascii=False, indent=2)
        
        # 保存文档元数据
        if doc_metadata is not None:
            with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(doc_metadata, f, ensure_ascii=False, indent=2)

        self.bm25.save(save_dir)

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
        sources_path = load_dir / "sources.json"
        metadata_path = load_dir / "metadata.json"

        if not vectors_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"向量数据库文件不存在: {load_dir}")

        self.vectors = np.load(vectors_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        if sources_path.exists():
            with open(sources_path, "r", encoding="utf-8") as f:
                self.sources = json.load(f)
        
        # 加载元数据（如果存在）
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                result = json.load(f)
        else:
            result = None

        self.bm25.load(load_dir)
        return result
