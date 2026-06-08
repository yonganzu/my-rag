"""
混合检索器：融合向量检索和 BM25 检索

使用 RRF（倒数排名融合）算法合并两个检索器的结果。

设计原则：
  - 将向量检索和关键词检索解耦
  - 使用统一的接口对外提供服务
  - 支持灵活的权重配置
"""

from typing import List, Tuple, Dict
import numpy as np

from src.config import config
from .base import VectorDB
from .bm25_retriever import BM25Retriever


class HybridRetriever:
    """
    混合检索器
    
    协调向量检索和 BM25 检索，使用 RRF 融合结果。
    
    使用示例：
        vector_db = VectorDBFactory.create("faiss")
        bm25_retriever = BM25Retriever()
        hybrid = HybridRetriever(vector_db, bm25_retriever)
        
        # 添加文档
        hybrid.add(chunks, vectors, sources)
        
        # 混合检索
        results = hybrid.hybrid_search(query_text, query_vector, top_k=3)
    """

    def __init__(self, vector_db: VectorDB, bm25_retriever: BM25Retriever):
        self.vector_db = vector_db
        self.bm25_retriever = bm25_retriever

    def add(self, chunks: List[str], vectors: np.ndarray, sources: List[str] = None) -> None:
        """添加文档块到两个检索器"""
        self.vector_db.add(chunks, vectors, sources)
        self.bm25_retriever.add_documents(chunks, sources)

    def build(self, chunks: List[str], vectors: np.ndarray, sources: List[str] = None) -> None:
        """构建索引（覆盖式）"""
        self.vector_db.add(chunks, vectors, sources)
        self.bm25_retriever.build(chunks, sources)

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """纯向量检索"""
        return self.vector_db.search(query_vector, top_k)

    def bm25_search(self, query_text: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """纯 BM25 检索"""
        results = self.bm25_retriever.search(query_text, top_k)
        return [(chunk, score, source) for _, score, chunk, source in results]

    def hybrid_search(
        self,
        query_text: str,
        query_vector: np.ndarray,
        top_k: int = 3,
        bm25_weight: float = 0.5,
    ) -> List[Tuple[str, float, str]]:
        """
        混合检索：向量语义 + BM25 关键词，用 RRF 融合
        
        参数：
          query_text: 查询文本（用于 BM25）
          query_vector: 查询向量（用于向量检索）
          top_k: 返回前 k 个结果
          bm25_weight: BM25 权重 (0~1)
        
        返回：
          [(chunk_text, rrf_score, source), ...]，按 RRF 分数从高到低排序
        """
        if self.vector_db.is_empty() or self.bm25_retriever.is_empty():
            return []

        RRF_K = config.rrf_k
        fetch_k = top_k * config.rrf_candidate_factor

        # -- 1. 向量检索 --
        vec_results = self.vector_db.search(query_vector, top_k=fetch_k)
        
        # 输出向量检索详细日志
        print(f"\n[向量检索] 共检索到 {len(vec_results)} 条结果")
        for i, (chunk, score, source) in enumerate(vec_results[:5]):  # 只显示前5条
            print(f"  [{i+1}] 相似度: {score:.4f} | 来源: {source}")
            print(f"      内容: {chunk[:80]}..." if len(chunk) > 80 else f"      内容: {chunk}")

        # -- 2. BM25 检索 --
        bm25_results = self.bm25_retriever.search(query_text, top_k=fetch_k)
        
        # 输出 BM25 检索详细日志
        print(f"\n[BM25 检索] 共检索到 {len(bm25_results)} 条结果")
        for i, (idx, score, chunk, source) in enumerate(bm25_results[:5]):  # 只显示前5条
            print(f"  [{i+1}] BM25分数: {score:.4f} | 来源: {source}")
            print(f"      内容: {chunk[:80]}..." if len(chunk) > 80 else f"      内容: {chunk}")

        # -- 3. 构建文档映射（用于去重和结果构建）--
        chunk_to_source: Dict[str, str] = {}
        for chunk, _, source in vec_results:
            chunk_to_source[chunk] = source
        for _, _, chunk, source in bm25_results:
            if chunk not in chunk_to_source:
                chunk_to_source[chunk] = source

        # -- 4. RRF 融合 --
        rrf_scores: Dict[str, float] = {}
        rrf_details: Dict[str, dict] = {}  # 存储详细分数信息

        # 向量检索的贡献
        print(f"\n[RRF 融合] RRF_K={RRF_K}, BM25权重={bm25_weight}")
        print("-" * 60)
        print(f"向量检索贡献（权重: {1.0 - bm25_weight:.2f}）:")
        
        for rank, (chunk, score, source) in enumerate(vec_results):
            vec_contribution = (1.0 - bm25_weight) / (RRF_K + rank + 1)
            rrf_scores[chunk] = rrf_scores.get(chunk, 0) + vec_contribution
            
            if chunk not in rrf_details:
                rrf_details[chunk] = {
                    'vec_rank': None,
                    'vec_score': None,
                    'vec_contribution': 0.0,
                    'bm25_rank': None,
                    'bm25_score': None,
                    'bm25_contribution': 0.0,
                    'total_score': 0.0,
                    'source': source
                }
            rrf_details[chunk]['vec_rank'] = rank + 1
            rrf_details[chunk]['vec_score'] = score
            rrf_details[chunk]['vec_contribution'] = vec_contribution
            rrf_details[chunk]['source'] = source
            
            if rank < 3:  # 只显示前3条的融合详情
                print(f"  排名 {rank+1}: vec贡献={vec_contribution:.6f} | 相似度={score:.4f}")

        # BM25 检索的贡献
        print(f"\nBM25 检索贡献（权重: {bm25_weight:.2f}）:")
        for rank, (idx, score, chunk, source) in enumerate(bm25_results):
            bm25_contribution = bm25_weight / (RRF_K + rank + 1)
            rrf_scores[chunk] = rrf_scores.get(chunk, 0) + bm25_contribution
            
            if chunk not in rrf_details:
                rrf_details[chunk] = {
                    'vec_rank': None,
                    'vec_score': None,
                    'vec_contribution': 0.0,
                    'bm25_rank': None,
                    'bm25_score': None,
                    'bm25_contribution': 0.0,
                    'total_score': 0.0,
                    'source': source
                }
            rrf_details[chunk]['bm25_rank'] = rank + 1
            rrf_details[chunk]['bm25_score'] = score
            rrf_details[chunk]['bm25_contribution'] = bm25_contribution
            rrf_details[chunk]['total_score'] = rrf_scores[chunk]
            
            if rank < 3:  # 只显示前3条的融合详情
                print(f"  排名 {rank+1}: BM25贡献={bm25_contribution:.6f} | BM25分数={score:.4f}")

        # -- 5. 按 RRF 分数排序 --
        sorted_chunks = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)

        # -- 6. 输出融合结果详情 --
        print("\n[融合结果详情] 按 RRF 分数排序:")
        print("-" * 80)
        print(f"{'排名':<4} {'向量排名':<8} {'向量分数':<10} {'BM25排名':<8} {'BM25分数':<10} {'RRF分数':<12} {'来源'}")
        print("-" * 80)
        
        for i, chunk in enumerate(sorted_chunks[:top_k]):
            details = rrf_details[chunk]
            vec_rank_str = str(details['vec_rank']) if details['vec_rank'] else '-'
            vec_score_str = f"{details['vec_score']:.4f}" if details['vec_score'] else '-'
            bm25_rank_str = str(details['bm25_rank']) if details['bm25_rank'] else '-'
            bm25_score_str = f"{details['bm25_score']:.4f}" if details['bm25_score'] else '-'
            
            print(f"{i+1:<4} {vec_rank_str:<8} {vec_score_str:<10} {bm25_rank_str:<8} {bm25_score_str:<10} {rrf_scores[chunk]:<12.6f} {details['source']}")

        # -- 7. 构建最终结果 --
        results = []
        for chunk in sorted_chunks[:top_k]:
            source = chunk_to_source.get(chunk, "未知来源")
            results.append((chunk, rrf_scores[chunk], source))

        print(f"\n[混合检索] 最终返回 {len(results)} 条结果")
        return results

    def save(self, dir_path: str, doc_metadata: dict = None) -> None:
        """保存两个检索器的数据"""
        self.vector_db.save(dir_path, doc_metadata)
        self.bm25_retriever.save(dir_path)

    def load(self, dir_path: str) -> dict | None:
        """加载两个检索器的数据"""
        metadata = self.vector_db.load(dir_path)
        self.bm25_retriever.load(dir_path)
        return metadata

    def __len__(self) -> int:
        """返回文档数量"""
        return len(self.vector_db)

    def is_empty(self) -> bool:
        """判断是否为空"""
        return self.vector_db.is_empty()
