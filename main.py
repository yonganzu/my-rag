"""
RAG 示例程序入口

为什么要单独一个 main.py？
  - 入口清晰，用户从这里运行程序
  - 将演示逻辑与业务逻辑分离（src/ 里的模块可被其他程序复用）
  - 遵循 Python 项目惯例：src/ 放核心代码，根目录放脚本

运行方式：
  export DASHSCOPE_API_KEY=sk-xxxx     # 先设好环境变量
  python main.py                        # 运行 RAG 示例
"""

from pathlib import Path

from src.config import config
from src.document_loader import load_documents_from_folder
from src.rag_pipeline import RAGPipeline


def main():
    # -- 1. 检查本地向量数据库 --
    rag = RAGPipeline()
    vector_db_path = rag.db_path
    
    if vector_db_path.exists() and any(vector_db_path.iterdir()):
        print(f"[发现本地向量数据库] {vector_db_path}")
        print("[直接加载已有知识库，跳过文档加载和向量化...]")
        rag.load_knowledge_base()
        chunks = None
    else:
        # -- 2. 加载文档并分块 --
        docs_folder = config.data_dir / "documents"
        print(f"[加载文档] 从文件夹: {docs_folder}\n")

        chunks = load_documents_from_folder(
            folder_path=str(docs_folder),
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
        )
        print(f"\n[OK] 共加载 {len(chunks)} 个文本块\n")

        # -- 3. 构建知识库 --
        rag.build_knowledge_base(chunks)

    # -- 3. 交互式问答 --
    use_rerank = config.use_rerank
    use_query_rewrite = config.use_query_rewrite

    print("\n" + "=" * 60)
    print("[RAG 问答系统已就绪！（输入 'exit' 退出）]")
    if use_rerank:
        print("[已启用语义重排序，检索结果将通过 LLM 重新排序]")
    if use_query_rewrite:
        print("[已启用 Query 改写，用户问题将通过 LLM 优化]")
    print("=" * 60)

    while True:
        question = input("\n[请输入问题]: ").strip()
        if question.lower() in ("exit", "quit", "q"):
            print("[再见！]")
            break
        if not question:
            continue

        try:
            print("[正在检索和生成回答...]")
            answer, contexts = rag.answer(question, use_rerank=use_rerank, use_query_rewrite=use_query_rewrite)
            
            # 显示检索到的上下文
            print("\n" + "-" * 60)
            print("[检索到的相关文档]:")
            print("-" * 60)
            for i, ctx in enumerate(contexts, 1):
                print(f"\n[{i}] {ctx[:200]}..." if len(ctx) > 200 else f"\n[{i}] {ctx}")
            
            # 显示回答
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
