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
from src.document_loader import load_and_chunk
from src.rag_pipeline import RAGPipeline


def main():
    # ── 1. 加载文档并分块 ──────────────────────────────────────
    doc_path = config.data_dir / "sample_document.txt"
    print(f"📄 加载文档: {doc_path}\n")

    chunks = load_and_chunk(
        file_path=str(doc_path),
        chunk_size=config.chunk_size,
        overlap=config.chunk_overlap,
    )
    print(f"✅ 文档被分割为 {len(chunks)} 个文本块\n")

    # ── 2. 构建知识库 ──────────────────────────────────────────
    rag = RAGPipeline()
    rag.build_knowledge_base(chunks)

    # ── 3. 交互式问答 ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("💡 RAG 问答系统已就绪！（输入 'exit' 退出）")
    print("=" * 60)

    while True:
        question = input("\n❓ 请输入问题: ").strip()
        if question.lower() in ("exit", "quit", "q"):
            print("👋 再见！")
            break
        if not question:
            continue

        try:
            print("🔍 正在检索和生成回答...")
            answer = rag.answer(question)
            print(f"\n🤖 回答:\n{answer}")
        except Exception as e:
            print(f"❌ 出错: {e}")


if __name__ == "__main__":
    # __name__ == "__main__" 确保只有直接运行此文件时才执行 main()
    # 当被 import 时不会自动运行，这是 Python 的标准实践
    main()
