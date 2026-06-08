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

from dashscope import Generation

from src.config import config
from src.embedding import embed_text
from src.vector_db import VectorDB, HybridRetriever, BM25Retriever


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
            self.bm25_retriever = BM25Retriever()
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

    def retrieve(
        self,
        question: str,
        use_rerank: bool = False,
        use_query_rewrite: bool = False,
    ) -> tuple[List[str], List[str]]:
        """
        检索与问题最相关的文档块

        返回：
          (contexts, sources) - 上下文列表和对应的来源文件名列表
        """
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
            results = self.hybrid_retriever.hybrid_search(
                query_text=question,
                query_vector=query_vector,
                top_k=fetch_k,
                bm25_weight=config.bm25_weight,
            )
        else:
            results = self.vector_db.search(query_vector, top_k=fetch_k)

        # 拆分为文本和来源
        contexts = [chunk for chunk, score, source in results]
        sources = [source for chunk, score, source in results]

        print(f"检索到 {len(contexts)} 个候选文档块")

        # 使用 LLM 进行重排序
        if use_rerank and len(contexts) > 1:
            contexts, sources = self.rerank(
                original_question if use_query_rewrite else question,
                contexts,
                sources,
            )

        return contexts[:config.top_k], sources[:config.top_k]

    def rewrite_query(self, question: str) -> str:
        """使用 LLM 对用户查询进行改写"""
        print("[正在进行 Query 改写...]")

        prompt = build_query_rewrite_prompt(question)

        resp = Generation.call(
            model=config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            api_key=config.dashscope_api_key,
            result_format="message",
        )

        if resp.status_code != 200:
            print(f"Query 改写 API 调用失败，使用原始问题: {resp.message}")
            return question

        try:
            result = resp.output.choices[0].message.content.strip()
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

    def rerank(
        self,
        question: str,
        contexts: List[str],
        sources: List[str] = None,
    ) -> tuple[List[str], List[str]]:
        """使用 LLM 对检索结果进行重排序"""
        print("[正在进行语义重排序...]")

        prompt = build_rerank_prompt(question, contexts)

        resp = Generation.call(
            model=config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            api_key=config.dashscope_api_key,
            result_format="message",
        )

        if resp.status_code != 200:
            print(f"Rerank API 调用失败，使用原始排序: {resp.message}")
            return contexts, sources or []

        try:
            result = resp.output.choices[0].message.content.strip()
            ranks = [int(x.strip()) - 1 for x in result.split(",")]

            if len(ranks) != len(contexts) or set(ranks) != set(range(len(contexts))):
                raise ValueError("排序结果无效")

            reranked_contexts = [contexts[i] for i in ranks]
            reranked_sources = [sources[i] for i in ranks] if sources else []
            print("[OK] 语义重排序完成")
            return reranked_contexts, reranked_sources
        except Exception as e:
            print(f"解析排序结果失败，使用原始排序: {e}")
            return contexts, sources or []
