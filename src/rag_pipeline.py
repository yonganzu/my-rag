"""
RAG（Retrieval-Augmented Generation）流水线

这是整个应用的核心，串联了前面所有模块：

  用户问题
     │
     ▼
  ┌─────────────┐
  │  Embedding   │  将问题转为向量
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  向量检索    │  在知识库中找到最相关的文档块
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  构建 Prompt │  问题 + 检索到的上下文 → 结构化提示词
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  LLM 生成   │  调用通义千问生成最终回答
  └─────────────┘

为什么要用 RAG 而不是直接问 LLM？
  - LLM 的知识有截止日期，不知道最新信息
  - LLM 不知道你的私有文档/数据库里的内容
  - RAG 让 LLM "先读资料再回答"，减少幻觉（hallucination）

为什么叫 "流水线"（Pipeline）？
  - 数据像流水线一样依次经过每个处理阶段
  - 每个阶段职责单一、可独立测试和替换
  - 工程中常见的模式（如数据管道 ETL Pipeline）
"""

from typing import List, Optional

from dashscope import Generation

from src.config import config
from src.embedding import embed_texts, embed_text
from src.vector_store import VectorStore


def build_rag_prompt(question: str, contexts: List[str]) -> str:
    """
    构建 RAG 提示词

    结构：
      1. 系统指令：告诉 LLM 如何行为
      2. 上下文：检索到的相关文档
      3. 用户问题

    为什么这样设计 prompt？
      - 先给上下文再给问题，LLM 更容易聚焦
      - 明确告诉 LLM "不知道就说不知道"，减少幻觉
      - 引用来源增加了回答的可信度（生产环境可加 citation）
    """
    # 用分隔线清晰地区分不同文档块
    context_text = "\n---\n".join(contexts)

    prompt = f"""你是一个基于知识库的问答助手。请根据以下上下文内容回答用户的问题。

上下文：
{context_text}

请基于以上上下文回答问题。如果上下文中没有足够信息，请如实说"根据提供的资料无法回答这个问题"，不要编造。

用户问题：{question}
"""
    return prompt


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


class RAGPipeline:
    """
    RAG 流水线，管理知识库构建和问答的全流程

    使用方式：
      rag = RAGPipeline()
      rag.build_knowledge_base(chunks)   # 第一步：建立知识库（自动保存）
      answer = rag.answer("你的问题")     # 第二步：提问

    持久化：
      - 知识库会自动保存到 data/vector_db 目录
      - 下次运行时可以直接加载：rag.load_knowledge_base()
    """

    def __init__(self, db_path: str = "data/vector_db"):
        from pathlib import Path
        self.db_path = Path(db_path)
        self.vector_store = VectorStore()
        self._ready = False

    def build_knowledge_base(self, chunks: List[str]) -> None:
        """
        构建知识库：将文档块全部转成向量并存入向量存储

        为什么不逐条 embed？
          - batch embedding 比逐条调用快 10 倍以上
          - dashscope SDK 内部会并发请求
        """
        print(f"正在对 {len(chunks)} 个文档块生成向量嵌入...")

        vectors = embed_texts(
            texts=chunks,
            model=config.embedding_model,
        )

        self.vector_store.add(chunks, vectors)
        self._ready = True
        print(f"知识库构建完成，共 {len(self.vector_store)} 个文档块")

        self.vector_store.save(self.db_path)
        print(f"知识库已保存到 {self.db_path}")

    def load_knowledge_base(self) -> None:
        """从磁盘加载已构建的知识库"""
        self.vector_store.load(self.db_path)
        self._ready = True
        print(f"知识库已从 {self.db_path} 加载，共 {len(self.vector_store)} 个文档块")

    def retrieve(self, question: str, use_rerank: bool = False, use_query_rewrite: bool = False) -> List[str]:
        """
        检索与问题最相关的文档块
        
        参数：
          question: 用户问题
          use_rerank: 是否使用 LLM 进行重排序
          use_query_rewrite: 是否使用 Query 改写
        
        返回：
          排序后的上下文列表
        """
        if not self._ready:
            raise RuntimeError("知识库尚未构建，请先调用 build_knowledge_base()")

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

        # 在向量存储中检索
        # 如果使用 rerank，先获取更多候选
        fetch_k = config.top_k * config.rerank_factor if use_rerank else config.top_k
        results = self.vector_store.search(query_vector, top_k=fetch_k)

        # 只返回文本内容，丢弃分数
        contexts = [chunk for chunk, score in results]

        print(f"检索到 {len(contexts)} 个候选文档块")

        # 使用 LLM 进行重排序
        if use_rerank and len(contexts) > 1:
            contexts = self.rerank(original_question if use_query_rewrite else question, contexts)

        # 最终只返回 top_k 个
        return contexts[:config.top_k]

    def rewrite_query(self, question: str) -> str:
        """
        使用 LLM 对用户查询进行改写
        
        参数：
          question: 原始用户问题
        
        返回：
          改写后的问题
        """
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
            
            # 解析改写后的问题
            lines = result.split("\n")
            rewritten_question = question
            
            for line in lines:
                if line.startswith("改写问题：") or line.startswith("改写问题:"):
                    rewritten_question = line.split("：")[1].strip() if "：" in line else line.split(":")[1].strip()
                    break
            
            print(f"[OK] Query 改写完成")
            return rewritten_question
        except Exception as e:
            print(f"解析改写结果失败，使用原始问题: {e}")
            return question

    def rerank(self, question: str, contexts: List[str]) -> List[str]:
        """
        使用 LLM 对检索结果进行重排序
        
        参数：
          question: 用户问题
          contexts: 待排序的上下文列表
        
        返回：
          重排序后的上下文列表
        """
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
            return contexts

        try:
            # 解析排序结果
            result = resp.output.choices[0].message.content.strip()
            ranks = [int(x.strip()) - 1 for x in result.split(",")]
            
            # 验证排序结果
            if len(ranks) != len(contexts) or set(ranks) != set(range(len(contexts))):
                raise ValueError("排序结果无效")
            
            # 重新排序
            reranked_contexts = [contexts[i] for i in ranks]
            print("[OK] 语义重排序完成")
            return reranked_contexts
        except Exception as e:
            print(f"解析排序结果失败，使用原始排序: {e}")
            return contexts

    def generate(self, question: str, contexts: List[str]) -> str:
        """根据问题和上下文生成回答"""
        prompt = build_rag_prompt(question, contexts)

        # ── 调用 DashScope Generation API ──────────────────────
        # 兼容 OpenAI 格式，也可以用 OpenAI SDK 调用
        resp = Generation.call(
            model=config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            api_key=config.dashscope_api_key,
            result_format="message",  # 返回结构化消息格式
        )

        if resp.status_code != 200:
            raise RuntimeError(
                f"LLM API 调用失败: [{resp.status_code}] {resp.message}"
            )

        return resp.output.choices[0].message.content

    def answer(self, question: str, use_rerank: bool = False, use_query_rewrite: bool = False) -> tuple[str, list[str]]:
        """
        一站式问答：[Query 改写] → 检索 → [重排序] → 生成

        这就是 RAG 最核心的 "检索增强生成" 模式
        
        参数：
          question: 用户问题
          use_rerank: 是否使用语义重排序
          use_query_rewrite: 是否使用 Query 改写
        
        返回：
          (answer, contexts) - 回答内容和检索到的上下文列表
        """
        contexts = self.retrieve(question, use_rerank=use_rerank, use_query_rewrite=use_query_rewrite)
        answer = self.generate(question, contexts)
        return answer, contexts
