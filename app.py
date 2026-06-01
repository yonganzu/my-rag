"""
RAG 知识库问答系统 - ChatGPT 风格界面
"""

import os
import gradio as gr
from src.config import config
from src.document_loader import load_documents_from_folder
from src.rag_pipeline import RAGPipeline
import shutil
from pathlib import Path

rag = None
knowledge_base_loaded = False
upload_folder = config.data_dir / "documents"
upload_folder.mkdir(parents=True, exist_ok=True)

CSS = """
/* 整体风格：仿 ChatGPT 简洁白色调 */
.gradio-container {
    max-width: 100% !important;
}
.sidebar {
    background: #f7f7f8 !important;
    border-right: 1px solid #e5e5e5 !important;
    min-height: 100vh;
    padding: 20px 16px !important;
}
.main-chat {
    background: #ffffff !important;
    min-height: 100vh;
}
.doc-item {
    padding: 10px 12px;
    margin: 4px 0;
    border-radius: 8px;
    background: #ffffff;
    border: 1px solid #e5e5e5;
    font-size: 14px;
    transition: background 0.2s;
}
.doc-item:hover {
    background: #f0f0f0;
}
.doc-name {
    font-weight: 600;
    color: #202123;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.doc-info {
    font-size: 12px;
    color: #8e8ea0;
    margin-top: 2px;
}
.sidebar-header {
    font-size: 16px;
    font-weight: 700;
    color: #202123;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid #e5e5e5;
}
.empty-state {
    text-align: center;
    color: #8e8ea0;
    font-size: 13px;
    padding: 24px 0;
}
.chat-footer {
    position: sticky;
    bottom: 0;
    background: #ffffff;
    padding: 12px 0;
    border-top: 1px solid #f0f0f0;
}
footer {
    display: none !important;
}
"""


def format_size(size_bytes: float) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def get_document_list():
    """获取向量库中的文档列表"""
    global rag

    items = []

    if rag is not None and rag._doc_metadata is not None and len(rag._doc_metadata) > 0:
        items = []
        for fname, meta in rag._doc_metadata.items():
            size_str = format_size(meta.get("size", 0))
            items.append(
                f'<div class="doc-item"><div class="doc-name">📄 {fname}</div>'
                f'<div class="doc-info">{size_str}</div></div>'
            )
    elif upload_folder.exists():
        found = False
        supported_exts = {".txt", ".docx", ".xlsx", ".pptx", ".pdf", ".html"}
        for f in sorted(upload_folder.iterdir()):
            if f.is_file() and f.suffix.lower() in supported_exts:
                found = True
                size_str = format_size(f.stat().st_size)
                items.append(
                    f'<div class="doc-item"><div class="doc-name">📄 {f.name}</div>'
                    f'<div class="doc-info">{size_str} (未入库)</div></div>'
                )
        if not found:
            return '<div class="empty-state">暂无文档<br>请上传文档构建知识库</div>'

    if not items:
        return '<div class="empty-state">暂无文档<br>请上传文档构建知识库</div>'

    return "\n".join(items)


def get_kb_stats():
    """获取知识库统计信息"""
    global rag, knowledge_base_loaded
    if rag is not None and knowledge_base_loaded:
        chunk_count = len(rag.vector_store)
        doc_count = len(rag._doc_metadata) if rag._doc_metadata else 0
        return f"📊 {doc_count} 个文档 · {chunk_count} 个文本块"
    return "📊 知识库未加载"


def load_kb():
    global rag, knowledge_base_loaded
    if rag is None:
        rag = RAGPipeline()

    vector_db_path = rag.db_path
    if vector_db_path.exists() and any(vector_db_path.iterdir()):
        rag.load_knowledge_base()
        has_update, new_files, modified_files = rag.check_documents_update(upload_folder)

        if has_update:
            updated_files = new_files + modified_files
            chunks, doc_metadata = load_documents_from_folder(
                folder_path=str(upload_folder),
                chunk_size=config.chunk_size,
                overlap=config.chunk_overlap,
                specific_files=updated_files,
            )
            if chunks:
                rag.add_documents(chunks, doc_metadata)

        knowledge_base_loaded = True
    else:
        knowledge_base_loaded = False

    return get_document_list(), get_kb_stats()


def process_upload(files):
    global rag, knowledge_base_loaded
    if not files:
        return get_document_list(), get_kb_stats(), "请选择文件"

    uploaded_names = []
    for file in files:
        if isinstance(file, str):
            src = Path(file)
            dst = upload_folder / src.name
            shutil.copy2(src, dst)
            uploaded_names.append(src.name)
        elif hasattr(file, 'name'):
            dst = upload_folder / file.name
            shutil.copy2(file.name, dst)
            uploaded_names.append(file.name)

    if knowledge_base_loaded and rag is not None:
        chunks, doc_metadata = load_documents_from_folder(
            folder_path=str(upload_folder),
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
            specific_files=uploaded_names,
        )
        rag.add_documents(chunks, doc_metadata)
        msg = f"✅ 已添加: {', '.join(uploaded_names)}"
    else:
        chunks, doc_metadata = load_documents_from_folder(
            folder_path=str(upload_folder),
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
        )
        rag.build_knowledge_base(chunks, doc_metadata)
        knowledge_base_loaded = True
        msg = f"✅ 知识库已构建 ({len(chunks)} 个文本块)"

    return get_document_list(), get_kb_stats(), msg


def chat(message, history):
    if not knowledge_base_loaded or rag is None:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "⚠️ 请先在左侧上传文档构建知识库"})
        return history

    try:
        answer, contexts = rag.answer(
            message,
            use_rerank=config.use_rerank,
            use_query_rewrite=config.use_query_rewrite,
        )
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})
        return history
    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"❌ 错误: {str(e)}"})
        return history


def clear_chat():
    return []


def delete_all_docs():
    global rag, knowledge_base_loaded
    if upload_folder.exists():
        supported_exts = {".txt", ".docx", ".xlsx", ".pptx", ".pdf", ".html"}
        for f in upload_folder.iterdir():
            if f.is_file() and f.suffix.lower() in supported_exts:
                f.unlink()
    if rag is not None and rag.vector_store is not None:
        rag.vector_store.chunks = []
        rag.vector_store.vectors = None
        rag._doc_metadata = None
        rag._ready = False
    knowledge_base_loaded = False
    db_path = Path("data/vector_db")
    if db_path.exists():
        import shutil as _shutil
        _shutil.rmtree(db_path)
    return get_document_list(), get_kb_stats(), None, "🗑️ 知识库已清空"


with gr.Blocks(
    title="RAG 知识库问答",
    css=CSS,
) as demo:

    with gr.Row(equal_height=True):
        # ========== 左侧：文档管理 ==========
        with gr.Column(scale=1, elem_classes="sidebar"):
            gr.HTML('<div class="sidebar-header">📚 知识库文档</div>')

            kb_stats = gr.Textbox(
                value="📊 加载中...",
                label=None,
                interactive=False,
                container=False,
                elem_classes="kb-stats",
            )

            doc_list = gr.HTML(value='<div class="empty-state">加载中...</div>')

            upload_btn = gr.UploadButton(
                "📁 上传文档",
                file_count="multiple",
                variant="primary",
                size="sm",
            )

            delete_btn = gr.Button(
                "🗑️ 清空知识库",
                variant="stop",
                size="sm",
            )

            upload_msg = gr.Textbox(
                label=None,
                interactive=False,
                container=False,
                visible=True,
                show_label=False,
            )

        # ========== 右侧：对话区域 ==========
        with gr.Column(scale=3, elem_classes="main-chat"):
            gr.HTML(
                '<div style="text-align:center;padding:12px 0 8px 0;'
                'font-size:18px;font-weight:700;color:#202123;">'
                '💬 RAG 知识库问答</div>'
            )

            chatbot = gr.Chatbot(
                value=[],
                height=520,
                show_label=False,
            )

            with gr.Row(elem_classes="chat-footer"):
                msg_input = gr.Textbox(
                    placeholder="输入你的问题，按 Enter 发送...",
                    show_label=False,
                    scale=8,
                    container=False,
                )
                send_btn = gr.Button(
                    "发送",
                    variant="primary",
                    scale=1,
                    size="sm",
                )
                clear_btn = gr.Button(
                    "清空对话",
                    variant="secondary",
                    scale=1,
                    size="sm",
                )

    # ========== 事件绑定 ==========
    upload_btn.upload(
        process_upload,
        upload_btn,
        [doc_list, kb_stats, upload_msg],
    )

    send_btn.click(
        chat,
        [msg_input, chatbot],
        chatbot,
    ).then(lambda: "", None, msg_input)

    msg_input.submit(
        chat,
        [msg_input, chatbot],
        chatbot,
    ).then(lambda: "", None, msg_input)

    clear_btn.click(
        clear_chat,
        None,
        chatbot,
    )

    delete_btn.click(
        delete_all_docs,
        None,
        [doc_list, kb_stats, chatbot, upload_msg],
    )

    demo.load(
        load_kb,
        None,
        [doc_list, kb_stats],
    )

if __name__ == "__main__":
    print("启动 RAG 知识库问答系统...")
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860)
