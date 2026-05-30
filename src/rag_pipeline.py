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

    def retrieve(self, question: str) -> List[str]:
        """检索与问题最相关的文档块"""
        if not self._ready:
            raise RuntimeError("知识库尚未构建，请先调用 build_knowledge_base()")

        # 将问题转为向量
        query_vector = embed_text(
            text=question,
            model=config.embedding_model,
        )

        # 在向量存储中检索
        results = self.vector_store.search(query_vector, top_k=config.top_k)

        # 只返回文本内容，丢弃分数
        contexts = [chunk for chunk, score in results]

        print(f"检索到 {len(contexts)} 个相关文档块")
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

    def answer(self, question: str) -> str:
        """
        一站式问答：检索 → 生成

        这就是 RAG 最核心的 "检索增强生成" 模式
        """
        contexts = self.retrieve(question)
        answer = self.generate(question, contexts)
        return answer
