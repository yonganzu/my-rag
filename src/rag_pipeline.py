"""
RAG（Retrieval-Augmented Generation）流水线

这是整个应用的核心，串联了前面所有模块：

  用户问题
     │
     ▼
  ┌─────────────┐
  │  Retrieval   │  Query 改写 → Embedding → 混合检索 → Rerank
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

模块分工：
  - retrieval.py  →  检索逻辑（向量/BM25/混合/改写/重排序）
  - embedding.py  →  文本向量化
  - vector_store.py → 向量存储与索引
  - 本文件        →  流水线编排 + Prompt 构建 + 生成
"""

from typing import List, Optional, Tuple
from pathlib import Path

from typing import Optional as _Optional

from src.config import config
from src.embedding import embed_texts
from src.llm import llm_call
from src.vector_db import VectorDBFactory
from src.retrieval import Retriever
from src.query_rewriter import rewrite_query


def build_rag_prompt(
    question: str,
    contexts: List[str],
    sources: List[str] = None,
    show_citations: bool = False,
    conversation_history: str = "",
) -> str:
    """构建 RAG 提示词"""
    # 对话历史
    history_section = ""
    if conversation_history:
        history_section = f"## 对话历史：\n{conversation_history}\n\n"

    if show_citations and sources:
        context_parts = []
        for ctx, src in zip(contexts, sources):
            context_parts.append(f"[来源: {src}]\n{ctx}")
        context_text = "\n---\n".join(context_parts)

        prompt = f"""你是一个基于知识库的问答助手。请根据以下上下文内容回答用户的问题。

{history_section}上下文：
{context_text}

请基于以上上下文回答问题。如果上下文中没有足够信息，请如实说"根据提供的资料无法回答这个问题"，不要编造。

请在回答中引用信息来源。如果你的观点参考了上下文中的特定内容，请注明"【参考：文件名】"。

用户问题：{question}
"""
    else:
        context_text = "\n---\n".join(contexts)

        prompt = f"""你是一个基于知识库的问答助手。请根据以下上下文内容回答用户的问题。

{history_section}上下文：
{context_text}

请基于以上上下文回答问题。如果上下文中没有足够信息，请如实说"根据提供的资料无法回答这个问题"，不要编造。

用户问题：{question}
"""
    return prompt


class RAGPipeline:
    """
    RAG 流水线：负责知识库管理和问答编排

    后端自闭环：注入 ConversationManager 后，answer() 只需传 user_id + conversation_id，
    上下文获取、query改写、检索、生成全部在后端完成，前端只负责展示。

    检索逻辑已拆分到 retrieval.Retriever，本类专注于：
      - 知识库的构建 / 加载 / 增量更新 / 变更检测
      - 对话上下文获取与管理
      - Prompt 构建与 LLM 生成
      - 将检索 + 生成串联为 complete 的问答流程
    
    支持的向量数据库后端：
      - faiss: FAISS（推荐，高性能）
      - memory: 内存实现（降级方案）
    """

    def __init__(
        self,
        db_path: str = "data/vector_db",
        db_type: str = None,
        conversation_manager: _Optional[object] = None,
    ):
        self.db_path = Path(db_path)
        self.vector_db = VectorDBFactory.create(db_type or config.vector_db_type)
        self.retriever = Retriever(self.vector_db)
        self._ready = False
        self._doc_metadata = {}
        self._conversation_manager = conversation_manager  # 可选的对话管理器注入

    # ═════════════════════════════════════════════════════════════
    # 知识库管理
    # ═════════════════════════════════════════════════════════════

    def build_knowledge_base(
        self,
        chunks: List[str],
        doc_metadata: dict = None,
        chunk_sources: List[str] = None,
    ) -> None:
        print(f"正在对 {len(chunks)} 个文档块生成向量嵌入...")

        vectors = embed_texts(texts=chunks, model=config.embedding_model)

        self.retriever.build_index(chunks, vectors, chunk_sources)
        self._ready = True
        self._doc_metadata = doc_metadata or {}
        print(f"知识库构建完成，共 {len(self.vector_db)} 个文档块")

        self.vector_db.save(self.db_path, self._doc_metadata)
        # 同时保存 BM25 索引
        self.retriever.save(str(self.db_path))
        print(f"知识库已保存到 {self.db_path}")

    def load_knowledge_base(self) -> bool:
        try:
            metadata = self.vector_db.load(self.db_path)
            self._doc_metadata = metadata if metadata else {}
            # 同时加载 BM25 索引
            self.retriever.load(str(self.db_path))
            self._ready = True
            print(f"知识库已从 {self.db_path} 加载，共 {len(self.vector_db)} 个文档块")
            return True
        except FileNotFoundError:
            print(f"向量数据库文件不存在: {self.db_path}")
            return False

    def check_documents_update(
        self, docs_folder: str | Path
    ) -> tuple[bool, list[str], list[str]]:
        docs_folder = Path(docs_folder)
        current_metadata = {}
        for file in docs_folder.rglob("*"):
            if file.is_file():
                current_metadata[file.name] = {
                    "mtime": file.stat().st_mtime,
                    "size": file.stat().st_size,
                }

        # 如果没有旧的元数据或元数据为空，所有文件都视为新增
        if self._doc_metadata is None or len(self._doc_metadata) == 0:
            return True, list(current_metadata.keys()), []

        old_files = set(self._doc_metadata.keys())
        current_files = set(current_metadata.keys())
        new_files = list(current_files - old_files)
        deleted_files = list(old_files - current_files)

        modified_files = []
        for file_name in old_files & current_files:
            old_info = self._doc_metadata.get(file_name, {})
            cur_info = current_metadata[file_name]
            if old_info.get("mtime") != cur_info["mtime"] or old_info.get("size") != cur_info["size"]:
                modified_files.append(file_name)

        has_update = len(new_files) > 0 or len(deleted_files) > 0 or len(modified_files) > 0

        if has_update:
            if new_files:
                print(f"[检测到] 新增文件: {', '.join(new_files)}")
            if deleted_files:
                print(f"[检测到] 删除文件: {', '.join(deleted_files)}")
            if modified_files:
                print(f"[检测到] 修改文件: {', '.join(modified_files)}")

        return has_update, new_files, modified_files

    def add_documents(
        self,
        chunks: List[str],
        doc_metadata: dict = None,
        chunk_sources: List[str] = None,
    ) -> None:
        if len(chunks) == 0:
            print("[增量更新] 没有新增文档块")
            return

        print(f"[增量更新] 正在对 {len(chunks)} 个新文档块生成向量嵌入...")

        vectors = embed_texts(texts=chunks, model=config.embedding_model)
        self.retriever.add_documents(chunks, vectors, chunk_sources)

        # 合并元数据：确保 _doc_metadata 始终是字典，doc_metadata 可能为 None
        if doc_metadata is not None:
            self._doc_metadata.update(doc_metadata)

        print(f"[增量更新] 已添加 {len(chunks)} 个文档块，知识库共 {len(self.vector_db)} 个文档块")
        self.vector_db.save(self.db_path, self._doc_metadata)
        # 同时保存 BM25 索引
        self.retriever.save(str(self.db_path))
        print(f"[增量更新] 知识库已保存到 {self.db_path}")

    # ═════════════════════════════════════════════════════════════
    # 生成
    # ═════════════════════════════════════════════════════════════

    def generate(
        self,
        question: str,
        contexts: List[str],
        sources: List[str] = None,
        show_citations: bool = False,
        conversation_history: str = "",
    ) -> str:
        prompt = build_rag_prompt(question, contexts, sources, show_citations, conversation_history)

        messages = []
        if conversation_history:
            # 将对话历史作为已有的上下文传给 LLM
            messages.append({
                "role": "system",
                "content": "你是基于知识库的问答助手。请根据对话历史和参考上下文回答用户问题。注意结合历史上下文理解用户当前的提问意图。"
            })
        return llm_call(
            model=config.llm_model,
            messages=messages + [{"role": "user", "content": prompt}],
            api_key=config.dashscope_api_key,
            base_url=config.llm_base_url,
        )

    def _get_conversation_context(self, user_id: str, conversation_id: str) -> str:
        """
        从 ConversationManager 获取格式化的对话历史文本（后端自闭环）

        Args:
            user_id: 用户ID
            conversation_id: 对话ID

        Returns:
            str: 格式化的对话历史，如 "用户：rag\n助手：RAG是一种..."
        """
        if not self._conversation_manager or not user_id or not conversation_id:
            return ""

        try:
            raw = self._conversation_manager.get_conversation_messages(
                user_id, conversation_id, limit=10
            )
        except Exception as e:
            print(f"[上下文管理] 获取对话历史失败: {e}")
            return ""

        if not raw:
            return ""

        lines = []
        for m in raw[:-1] if len(raw) > 1 else []:
            role_name = "用户" if m["role"] == "user" else "助手"
            lines.append(f"{role_name}：{m['content']}")
        return "\n".join(lines) if lines else ""

    # ── setter ─────────────────────────────────────────────

    def set_conversation_manager(self, mgr: object) -> None:
        """注入对话管理器（晚绑定，避免循环导入）"""
        self._conversation_manager = mgr

    # ═════════════════════════════════════════════════════════════
    # 问答
    # ═════════════════════════════════════════════════════════════

    def answer(
        self,
        question: str,
        use_rerank: bool = False,
        use_query_rewrite: bool = False,
        show_citations: bool = False,
        user_role: str = "user",
        user_id: str = "",
        conversation_id: str = "",
        conversation_history: str = "",
    ) -> tuple[str, list[str]]:
        """
        后端自闭环问答接口

        前端只需传 question + user_id + conversation_id，后端自动获取上下文、
        改写query、检索、生成。也兼容直接传 conversation_history 字符串的旧调用方式。
        """
        # 优先使用后端自闭环获取上下文
        if not conversation_history and user_id and conversation_id:
            conversation_history = self._get_conversation_context(user_id, conversation_id)

        # 如果有对话历史，先改写query解决代词指代问题
        if conversation_history:
            resolved_question = rewrite_query(question, conversation_history)
            print(f"[上下文管理] 已获取对话上下文（{len(conversation_history)}字符）")
        else:
            resolved_question = question

        contexts, sources = self.retriever.retrieve(
            resolved_question,
            use_rerank=use_rerank,
            use_query_rewrite=use_query_rewrite,
            user_role=user_role,
        )
        answer = self.generate(resolved_question, contexts, sources, show_citations, conversation_history)
        return answer, contexts
