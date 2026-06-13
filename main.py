"""
RAG 示例程序入口

为什么要单独一个 main.py？
  - 入口清晰，用户从这里运行程序
  - 将演示逻辑与业务逻辑分离（src/ 里的模块可被其他程序复用）
  - 遵循 Python 项目惯例：src/ 放核心代码，根目录放脚本

运行方式：
  export DASHSCOPE_API_KEY=sk-xxxx     # 先设好环境变量
  python main.py                        # 运行 RAG 示例

命令行说明：
  - 直接输入问题进行问答
  - 输入 'new' 创建新会话
  - 输入 'list' 查看会话列表
  - 输入 'switch <id>' 切换会话
  - 输入 'delete <id>' 删除会话
  - 输入 'exit' 退出程序
"""

from pathlib import Path

from src.config import config
from src.document_loader import load_documents_from_folder
from src.rag_pipeline import RAGPipeline
from src.cli_conversation import CLIConversationManager, show_help


def main():
    # -- 1. 初始化 RAG 管道 --
    rag = RAGPipeline()
    vector_db_path = rag.db_path
    docs_folder = config.data_dir / "documents"
    
    # -- 2. 初始化命令行会话管理器 --
    cli_cm = CLIConversationManager()
    print(f"[会话] 已创建新会话: {cli_cm.current_conversation_id}")
    
    # -- 3. 检查本地向量数据库是否存在 --
    if vector_db_path.exists() and any(vector_db_path.iterdir()):
        print(f"[发现本地向量数据库] {vector_db_path}")
        
        # 加载已有知识库
        rag.load_knowledge_base()
        
        # -- 4. 检测文档是否有更新 --
        has_update, new_files, modified_files = rag.check_documents_update(docs_folder)
        
        if has_update:
            # 有更新，需要增量更新
            print("[文档有更新，进行增量更新...]")
            
            # 合并新增和修改的文件
            updated_files = new_files + modified_files
            
            # 加载更新的文档
            chunks, doc_metadata, chunk_sources = load_documents_from_folder(
                folder_path=str(docs_folder),
                chunk_size=config.chunk_size,
                overlap=config.chunk_overlap,
                specific_files=updated_files,
            )
            
            if chunks:
                rag.add_documents(chunks, doc_metadata, chunk_sources)
            else:
                print("[增量更新] 没有成功加载任何新文档")
        else:
            # 没有更新，直接使用已有知识库
            print("[文档无更新，直接使用已有知识库]")
            
    else:
        # -- 5. 首次构建：加载文档并分块 --
        print(f"[首次构建知识库] 从文件夹加载文档: {docs_folder}\n")

        chunks, doc_metadata, chunk_sources = load_documents_from_folder(
            folder_path=str(docs_folder),
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
        )
        print(f"\n[OK] 共加载 {len(chunks)} 个文本块\n")

        # -- 6. 构建知识库 --
        rag.build_knowledge_base(chunks, doc_metadata, chunk_sources)

    # -- 7. 交互式问答 --
    use_rerank = config.use_rerank
    use_query_rewrite = config.use_query_rewrite
    show_chunks = config.show_retrieved_chunks
    show_citations = config.show_citations

    print("\n" + "=" * 60)
    print("[RAG 问答系统已就绪！]")
    print(f"[当前会话: {cli_cm.current_conversation_id}]")
    if use_rerank:
        print("[已启用语义重排序]")
    if use_query_rewrite:
        print("[已启用 Query 改写]")
    if show_chunks:
        print("[将显示检索到的文档块]")
    if show_citations:
        print("[已启用引用来源]")
    if config.use_bm25:
        print(f"[已启用 BM25+向量混合检索]")
    print("[输入 'help' 查看更多命令]")
    print("=" * 60)

    while True:
        try:
            question = input(f"\n[{cli_cm.current_conversation_id}] 请输入问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[再见！]")
            break

        # 处理特殊命令
        cmd_lower = question.lower()
        
        if cmd_lower in ("exit", "quit", "q"):
            print("[再见！]")
            break
        
        if cmd_lower == "help":
            show_help()
            continue
        
        if cmd_lower == "new":
            new_id = cli_cm.new_conversation()
            print(f"[会话] 已创建新会话: {new_id}")
            continue
        
        if cmd_lower == "list":
            conversations = cli_cm.list_conversations()
            print("\n" + "=" * 50)
            print(f"[会话列表] (共 {len(conversations)} 个)")
            print("-" * 50)
            for conv in conversations:
                is_active = " [当前]" if conv["id"] == cli_cm.current_conversation_id else ""
                msg_count = len(conv.get("messages", []))
                print(f"  {conv['id']} | {conv['title'][:25]:<25} | {msg_count} 条消息{is_active}")
            print("=" * 50)
            print("提示: 使用 'switch <id>' 切换会话")
            continue
        
        if cmd_lower.startswith("switch "):
            parts = question.split()
            if len(parts) >= 2:
                success, conv = cli_cm.switch_conversation(parts[1])
                if success:
                    print(f"[会话] 已切换到: {cli_cm.current_conversation_id}")
                    # 显示该会话的历史消息
                    messages = conv.get("messages", [])
                    if messages:
                        print(f"[会话历史] 共 {len(messages)} 条消息:")
                        for msg in messages[-6:]:
                            role = "用户" if msg["role"] == "user" else "助手"
                            content = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
                            print(f"  [{role}] {content}")
                else:
                    print(f"[错误] 会话不存在: {parts[1]}")
            else:
                print("[用法] switch <会话ID>")
            continue
        
        if cmd_lower.startswith("delete "):
            parts = question.split()
            if len(parts) >= 2:
                if cli_cm.delete_conversation(parts[1]):
                    print(f"[会话] 已删除会话: {parts[1]}")
                    print(f"[会话] 当前会话: {cli_cm.current_conversation_id}")
                else:
                    print(f"[错误] 会话不存在: {parts[1]}")
            else:
                print("[用法] delete <会话ID>")
            continue
        
        if cmd_lower == "clear":
            new_id = cli_cm.new_conversation()
            print(f"[会话] 已创建新会话: {new_id}")
            continue
        
        if cmd_lower == "history":
            messages = cli_cm.get_history()
            if messages:
                print(f"\n[会话历史] 共 {len(messages)} 条消息:")
                for msg in messages:
                    role = "用户" if msg["role"] == "user" else "助手"
                    content = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
                    print(f"  [{role}] {content}")
            else:
                print("[会话] 当前会话暂无消息")
            continue
        
        if not question:
            continue

        try:
            print("[正在检索和生成回答...]")
            
            # 调用 RAG 回答
            answer, contexts = rag.answer(
                question,
                use_rerank=use_rerank,
                use_query_rewrite=use_query_rewrite,
                show_citations=show_citations,
            )

            # 保存消息
            cli_cm.send_message(question)
            cli_cm.save_response(answer)

            if config.show_retrieved_chunks:
                print("\n" + "-" * 60)
                print("[检索到的相关文档]:")
                print("-" * 60)
                for i, ctx in enumerate(contexts, 1):
                    print(f"\n[{i}] {ctx[:200]}..." if len(ctx) > 200 else f"\n[{i}] {ctx}")

            print("\n" + "=" * 60)
            print("[回答]:")
            print("=" * 60)
            print(answer)
        except Exception as e:
            print(f"[ERROR]: {e}")


if __name__ == "__main__":
    # __name__ == "__main__" 确保只有直接运行此文件时才执行 main()
    # 当被 import 时不会自动运行，这是 Python 的标准实践
    main()
