"""
检索模块

负责从知识库中检索与用户问题最相关的文档块。

检索策略：
  1. 向量检索（余弦相似度）：语义匹配
  2. BM25 关键词检索：精确词语匹配
  3. 混合检索（RRF 融合）：向量 + BM25 互补
  4. Query 改写：用 LLM 优化模糊/简短的问题
  5. Rerank 重排序：用 LLM 对候选结果重新排序

为什么把它们拆出来？
  - 检索是 RAG 中独立可替换的环节
  - 后续可以换成 Elasticsearch、FAISS 等而不用改 pipeline
  - 便于单独测试和调优
"""

from typing import List, Tuple, Optional

from src.config import config
from src.embedding import embed_text, embed_texts
from src.llm import llm_call
from src.vector_db import VectorDB, HybridRetriever, BM25Retriever
from src.permission import filter_docs_by_permission, get_user_max_doc_level, DocLevel
import numpy as np

# ── BGE-Reranker 模型缓存 ────────────────────────────────────────
_bge_reranker_model = None


def _get_bge_reranker(model_name: str = "BAAI/bge-reranker-v2-m3"):
    """获取 BGE-Reranker 模型（懒加载）"""
    global _bge_reranker_model
    if _bge_reranker_model is None:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            print(f"[Reranker] 加载本地 BGE-Reranker 模型: {model_name}")
            _bge_reranker_model = {
                "tokenizer": AutoTokenizer.from_pretrained(model_name),
                "model": AutoModelForSequenceClassification.from_pretrained(model_name),
            }
            print(f"[Reranker] BGE-Reranker 模型加载完成")
        except ImportError:
            raise ImportError("请安装 transformers 和 torch 以使用本地 BGE-Reranker: pip install transformers torch")
    return _bge_reranker_model


def build_rerank_prompt(question: str, contexts: List[str]) -> str:
    """
    构建 Rerank 提示词，用于对检索结果进行重排序

    Rerank 的作用：
      - 向量检索可能返回字面相似但语义不相关的结果
      - LLM 可以理解深层语义，进行更精准的排序
    """
    context_list = "\n".join([f"{i+1}. {ctx}" for i, ctx in enumerate(contexts)])
    
    prompt = f"""请对以下文档片段按照与用户问题的相关性进行排序。

用户问题：{question}

文档片段列表：
{context_list}

请按照相关性从高到低输出序号，用英文逗号分隔，不要输出其他内容。例如：3,1,2
"""
    return prompt


def build_query_rewrite_prompt(question: str) -> str:
    """
    构建 Query 改写提示词

    Query 改写的作用：
      - 将模糊/简短的问题改写为更清晰、更完整的表达
      - 补充隐含的上下文信息
      - 生成多个相关查询词，提升检索召回率
    """
    prompt = f"""请帮我优化以下用户问题，使其更适合文档检索。

用户问题：{question}

请提供：
1. 改写后的完整问题（更清晰、更具体）
2. 相关关键词列表

输出格式：
改写问题：[改写后的问题]
关键词：[关键词1, 关键词2, 关键词3]
"""
    return prompt


class Retriever:
    """
    检索器：负责从向量存储中检索相关文档块

    检索流程：
      Query 改写 → Embedding → 混合/向量检索 → Rerank
    """

    def __init__(self, vector_db: VectorDB):
        self.vector_db = vector_db
        self.hybrid_retriever: Optional[HybridRetriever] = None
        
        # 如果启用 BM25，创建混合检索器
        if config.use_bm25:
            self.bm25_retriever = BM25Retriever(k1=config.bm25_k1, b=config.bm25_b)
            self.hybrid_retriever = HybridRetriever(vector_db, self.bm25_retriever)

    def add_documents(self, chunks: List[str], vectors, sources: Optional[List[str]] = None) -> None:
        """添加文档到检索器（同时更新向量数据库和 BM25 索引）"""
        self.vector_db.add(chunks, vectors, sources)
        if self.hybrid_retriever:
            self.bm25_retriever.add_documents(chunks, sources)

    def build_index(self, chunks: List[str], vectors, sources: Optional[List[str]] = None) -> None:
        """构建索引（覆盖式）"""
        self.vector_db.add(chunks, vectors, sources)
        if self.hybrid_retriever:
            self.bm25_retriever.build(chunks, sources)

    def load(self, dir_path: str) -> None:
        """从磁盘加载索引（包括 BM25）"""
        if self.hybrid_retriever:
            success = self.bm25_retriever.load(dir_path)
            if success:
                print(f"[BM25] 索引加载成功，共 {len(self.bm25_retriever)} 个文档")
            else:
                print(f"[BM25] 索引加载失败或不存在，尝试从向量库重建...")
                # 从向量数据库中获取 chunks 和 sources 来重建 BM25
                if hasattr(self.vector_db, 'chunks') and hasattr(self.vector_db, 'sources'):
                    chunks = self.vector_db.chunks
                    sources = self.vector_db.sources if hasattr(self.vector_db, 'sources') else None
                    self.bm25_retriever.build(chunks, sources)
                    print(f"[BM25] 从向量库重建成功，共 {len(self.bm25_retriever)} 个文档")
                else:
                    print(f"[BM25] 无法从向量库重建，BM25 将保持为空")

    def save(self, dir_path: str) -> None:
        """保存索引到磁盘（包括 BM25）"""
        if self.hybrid_retriever:
            self.bm25_retriever.save(dir_path)
            print(f"[BM25] 索引已保存到 {dir_path}")

    def retrieve(
        self,
        question: str,
        use_rerank: bool = False,
        use_query_rewrite: bool = False,
        user_role: str = "user",
    ) -> tuple[List[str], List[str]]:
        """
        检索与问题最相关的文档块

        Args:
            question: 用户问题
            use_rerank: 是否使用重排序
            use_query_rewrite: 是否使用 Query 改写
            user_role: 用户角色，用于文档级别过滤

        返回：
          (contexts, sources) - 上下文列表和对应的来源文件名列表
        """
        # 获取用户可访问的最高文档级别
        max_doc_level = get_user_max_doc_level(user_role)
        print(f"[权限] 用户角色: {user_role}, 可访问最高级别: {max_doc_level}")
        
        # Query 改写
        original_question = question
        if use_query_rewrite:
            question = self.rewrite_query(question)
            print(f"[Query 改写] {original_question} -> {question}")

        # 将问题转为向量
        query_vector = embed_text(
            text=question,
            model=config.embedding_model,
        )

        fetch_k = config.top_k * config.rerank_factor if use_rerank else config.top_k

        if config.use_bm25 and self.hybrid_retriever:
            print(f"[混合检索] 向量语义 + BM25 关键词 (BM25 权重: {config.bm25_weight})")
            # 检查向量库和BM25是否为空
            vec_empty = self.vector_db.is_empty() if hasattr(self.vector_db, 'is_empty') else (len(self.vector_db) == 0)
            bm25_empty = self.bm25_retriever.is_empty() if hasattr(self.bm25_retriever, 'is_empty') else True
            
            print(f"[调试] 向量库状态: {'空' if vec_empty else f'有 {len(self.vector_db)} 个文档'}")
            print(f"[调试] BM25状态: {'空' if bm25_empty else f'有 {len(self.bm25_retriever)} 个文档'}")
            
            results = self.hybrid_retriever.hybrid_search(
                query_text=question,
                query_vector=query_vector,
                top_k=fetch_k,
                bm25_weight=config.bm25_weight,
            )
            print(f"[调试] 混合检索返回 {len(results)} 条结果")
        else:
            print(f"[向量检索] 仅使用向量数据库")
            vec_empty = self.vector_db.is_empty() if hasattr(self.vector_db, 'is_empty') else (len(self.vector_db) == 0)
            print(f"[调试] 向量库状态: {'空' if vec_empty else f'有 {len(self.vector_db)} 个文档'}")
            results = self.vector_db.search(query_vector, top_k=fetch_k)
            print(f"[调试] 向量检索返回 {len(results)} 条结果")

        # 拆分为文本和来源
        contexts = [chunk for chunk, score, source in results]
        sources = [source for chunk, score, source in results]

        print(f"检索到 {len(contexts)} 个候选文档块")
        
        # 根据用户角色过滤文档（硬控制）
        # 注意：这里假设文档来源中包含级别信息，格式为 "filename|level"
        # 如果没有级别信息，默认为 public
        if results:
            filtered_results = []
            for chunk, score, source in results:
                # 解析文档级别
                if "|" in source:
                    doc_source, doc_level = source.rsplit("|", 1)
                else:
                    doc_source = source
                    doc_level = DocLevel.PUBLIC.value
                
                # 检查用户是否有权限访问该文档
                from src.permission import DocLevel as DL
                if DL.can_access(max_doc_level, doc_level):
                    filtered_results.append((chunk, score, source))
            
            if len(filtered_results) < len(results):
                print(f"[权限过滤] 过滤了 {len(results) - len(filtered_results)} 个无权访问的文档")
            
            results = filtered_results
            contexts = [chunk for chunk, score, source in results]
            sources = [source for chunk, score, source in results]

        # 使用重排序
        if use_rerank and len(contexts) > 1:
            rerank_method = config.rerank_method
            print(f"[重排序] 使用方法: {rerank_method}")
            
            if rerank_method == "llm":
                # LLM 语义重排序（费用高）
                contexts, sources = self.rerank_with_llm(
                    original_question if use_query_rewrite else question,
                    contexts,
                    sources,
                )
            elif rerank_method == "vector":
                # 向量相似度重排序
                contexts, sources = self.rerank_with_vector(
                    original_question if use_query_rewrite else question,
                    contexts,
                    sources,
                )
            elif rerank_method == "keyword":
                # 关键词匹配重排序
                contexts, sources = self.rerank_with_keyword(
                    original_question if use_query_rewrite else question,
                    contexts,
                    sources,
                )
            elif rerank_method == "bge":
                # BGE-Reranker-v2-m3 重排序
                contexts, sources = self.rerank_with_bge(
                    original_question if use_query_rewrite else question,
                    contexts,
                    sources,
                )
            else:
                # 无重排序
                print("[重排序] 已跳过重排序")
        elif use_rerank:
            print("[重排序] 候选数量不足，跳过重排序")
        else:
            print("[重排序] 已禁用")

        return contexts[:config.top_k], sources[:config.top_k]

    def rewrite_query(self, question: str) -> str:
        """使用 LLM 对用户查询进行改写"""
        print("[正在进行 Query 改写...]")

        prompt = build_query_rewrite_prompt(question)

        try:
            result = llm_call(
                model=config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                api_key=config.dashscope_api_key,
                base_url=config.llm_base_url,
            )
        except Exception as e:
            print(f"Query 改写 API 调用失败，使用原始问题: {e}")
            return question

        try:
            lines = result.split("\n")
            rewritten_question = question
            for line in lines:
                if line.startswith("改写问题：") or line.startswith("改写问题:"):
                    rewritten_question = line.split("：")[1].strip() if "：" in line else line.split(":")[1].strip()
                    break
            print("[OK] Query 改写完成")
            return rewritten_question
        except Exception as e:
            print(f"解析改写结果失败，使用原始问题: {e}")
            return question

    def rerank_with_llm(
        self,
        question: str,
        contexts: List[str],
        sources: List[str] = None,
    ) -> tuple[List[str], List[str]]:
        """使用 LLM 对检索结果进行语义重排序（费用高）"""
        print("[重排序] 使用 LLM 语义重排序...")

        prompt = build_rerank_prompt(question, contexts)

        try:
            result = llm_call(
                model=config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                api_key=config.dashscope_api_key,
                base_url=config.llm_base_url,
            )
        except Exception as e:
            print(f"Rerank API 调用失败，使用原始排序: {e}")
            return contexts, sources or []

        try:
            ranks = [int(x.strip()) - 1 for x in result.split(",")]

            if len(ranks) != len(contexts) or set(ranks) != set(range(len(contexts))):
                raise ValueError("排序结果无效")

            reranked_contexts = [contexts[i] for i in ranks]
            reranked_sources = [sources[i] for i in ranks] if sources else []
            print("[重排序] LLM 语义重排序完成")
            return reranked_contexts, reranked_sources
        except Exception as e:
            print(f"解析排序结果失败，使用原始排序: {e}")
            return contexts, sources or []

    def rerank_with_vector(
        self,
        question: str,
        contexts: List[str],
        sources: List[str] = None,
    ) -> tuple[List[str], List[str]]:
        """使用向量相似度对检索结果进行重排序（免费）"""
        print("[重排序] 使用向量相似度重排序...")
        
        try:
            # 将问题和上下文都转为向量
            question_vector = embed_text(text=question, model=config.embedding_model)
            context_vectors = embed_texts(texts=contexts, model=config.embedding_model)
            
            # 确保向量是一维的
            if len(question_vector.shape) > 1:
                question_vector = question_vector.flatten()
            
            # 计算余弦相似度
            similarities = []
            for vec in context_vectors:
                # 确保向量是一维的
                if len(vec.shape) > 1:
                    vec = vec.flatten()
                
                # 计算余弦相似度
                norm_q = np.linalg.norm(question_vector)
                norm_c = np.linalg.norm(vec)
                
                if norm_q == 0 or norm_c == 0:
                    similarities.append(0.0)
                else:
                    sim = np.dot(question_vector, vec) / (norm_q * norm_c)
                    similarities.append(float(sim))
            
            # 按相似度排序
            indexed_sims = list(enumerate(similarities))
            indexed_sims.sort(key=lambda x: x[1], reverse=True)
            
            # 重排序
            reranked_contexts = [contexts[i] for i, _ in indexed_sims]
            reranked_sources = [sources[i] for i, _ in indexed_sims] if sources else []
            
            # 输出重排序详情
            print("[重排序] 向量相似度重排序结果:")
            for rank, (idx, sim) in enumerate(indexed_sims[:5]):
                print(f"  {rank+1}. 相似度: {sim:.4f} | {contexts[idx][:50]}...")
            
            print("[重排序] 向量相似度重排序完成")
            return reranked_contexts, reranked_sources
            
        except Exception as e:
            print(f"向量相似度重排序失败，使用原始排序: {e}")
            import traceback
            traceback.print_exc()
            return contexts, sources or []

    def rerank_with_keyword(
        self,
        question: str,
        contexts: List[str],
        sources: List[str] = None,
    ) -> tuple[List[str], List[str]]:
        """使用关键词匹配对检索结果进行重排序（免费）"""
        print("[重排序] 使用关键词匹配重排序...")
        
        try:
            # 分词
            import jieba
            question_keywords = set(jieba.cut(question))
            
            # 计算每个上下文的关键词匹配得分
            scores = []
            for context in contexts:
                context_keywords = set(jieba.cut(context))
                # 计算重叠关键词数量
                overlap = question_keywords & context_keywords
                # 归一化得分
                score = len(overlap) / max(len(question_keywords), 1)
                scores.append(score)
            
            # 按得分排序
            indexed_scores = list(enumerate(scores))
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 重排序
            reranked_contexts = [contexts[i] for i, _ in indexed_scores]
            reranked_sources = [sources[i] for i, _ in indexed_scores] if sources else []
            
            # 输出重排序详情
            print("[重排序] 关键词匹配重排序结果:")
            for rank, (idx, score) in enumerate(indexed_scores[:5]):
                overlap = question_keywords & set(jieba.cut(contexts[idx]))
                print(f"  {rank+1}. 匹配数: {len(overlap)} | 得分: {score:.4f} | {contexts[idx][:50]}...")
            
            print("[重排序] 关键词匹配重排序完成")
            return reranked_contexts, reranked_sources
            
        except Exception as e:
            print(f"关键词匹配重排序失败，使用原始排序: {e}")
            return contexts, sources or []

    def rerank_with_bge(
        self,
        question: str,
        contexts: List[str],
        sources: List[str] = None,
    ) -> tuple[List[str], List[str]]:
        """使用 BGE-Reranker-v2-m3 对检索结果进行重排序"""
        print(f"[重排序] 使用 BGE-Reranker-v2-m3 本地模式...")
        
        try:
            return self._rerank_with_bge_local(question, contexts, sources)
                
        except Exception as e:
            print(f"BGE-Reranker 重排序失败，使用原始排序: {e}")
            import traceback
            traceback.print_exc()
            return contexts, sources or []

    def _rerank_with_bge_local(
        self,
        question: str,
        contexts: List[str],
        sources: List[str] = None,
    ) -> tuple[List[str], List[str]]:
        """使用本地 BGE-Reranker 模型"""
        import torch
        
        reranker = _get_bge_reranker(config.bge_reranker_model)
        tokenizer = reranker["tokenizer"]
        model = reranker["model"]
        
        print(f"[Reranker] 使用本地模型: {config.bge_reranker_model}")
        
        # 准备输入对
        pairs = [[question, ctx] for ctx in contexts]
        
        # 批量处理
        scores = []
        batch_size = 8  # BGE-Reranker 更小的批量大小，内存占用更低
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            
            encoding = tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            
            with torch.no_grad():
                outputs = model(**encoding)
                logits = outputs.logits
                if logits.shape[1] >= 2:
                    # 对于二分类模型，取正类的概率
                    batch_scores = torch.sigmoid(logits[:, 1]).tolist()
                else:
                    # 对于回归模型或单输出模型
                    batch_scores = logits.squeeze().tolist()
                scores.extend(batch_scores)
        
        # 按分数排序
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 输出重排序详情
        print("[重排序] BGE-Reranker 本地重排序结果:")
        for rank, (idx, score) in enumerate(indexed_scores[:5]):
            print(f"  {rank+1}. 分数: {score:.4f} | {contexts[idx][:50]}...")
        
        # 重排序
        reranked_contexts = [contexts[i] for i, _ in indexed_scores]
        reranked_sources = [sources[i] for i, _ in indexed_scores] if sources else []
        
        print("[重排序] BGE-Reranker 本地重排序完成")
        return reranked_contexts, reranked_sources

    def rerank(
        self,
        question: str,
        contexts: List[str],
        sources: List[str] = None,
    ) -> tuple[List[str], List[str]]:
        """使用 LLM 对检索结果进行重排序（向后兼容）"""
        return self.rerank_with_llm(question, contexts, sources)
