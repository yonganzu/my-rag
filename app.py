"""
RAG 知识库问答系统 - 带用户登录和历史对话
"""

import os
import gradio as gr
from src.config import config
from src.document_loader import load_documents_from_folder
from src.rag_pipeline import RAGPipeline
from src.auth import auth_manager
from src.user_manager import UserManager
from src.conversation_manager import ConversationManager
import shutil
from pathlib import Path
from datetime import datetime

# 全局变量
rag = None
knowledge_base_loaded = False
upload_folder = config.data_dir / "documents"
upload_folder.mkdir(parents=True, exist_ok=True)

# 初始化管理器
user_manager = UserManager()
conversation_manager = ConversationManager()

# 当前登录用户
current_user = {"session_id": None, "username": None, "role": None}


def format_size(size_bytes: float) -> str:
    """格式化文件大小"""
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
            items.append(f"📄 {fname} ({size_str})")
    elif upload_folder.exists():
        supported_exts = {".txt", ".docx", ".xlsx", ".pptx", ".pdf", ".html"}
        for f in sorted(upload_folder.iterdir()):
            if f.is_file() and f.suffix.lower() in supported_exts:
                size_str = format_size(f.stat().st_size)
                items.append(f"📄 {f.name} ({size_str}) [未入库]")

    if not items:
        return "暂无文档"
    return "\n".join(items)


def get_kb_stats():
    """获取知识库统计信息"""
    global rag, knowledge_base_loaded
    if rag is not None and knowledge_base_loaded:
        chunk_count = len(rag.vector_db)
        doc_count = len(rag._doc_metadata) if rag._doc_metadata else 0
        return f"📊 {doc_count} 个文档 · {chunk_count} 个文本块"
    return "📊 知识库未加载"


def load_kb():
    """加载知识库"""
    global rag, knowledge_base_loaded
    if rag is None:
        rag = RAGPipeline()

    vector_db_path = rag.db_path
    if vector_db_path.exists() and any(vector_db_path.iterdir()):
        rag.load_knowledge_base()
        has_update, new_files, modified_files = rag.check_documents_update(upload_folder)

        if has_update:
            updated_files = new_files + modified_files
            chunks, doc_metadata, chunk_sources = load_documents_from_folder(
                folder_path=str(upload_folder),
                chunk_size=config.chunk_size,
                overlap=config.chunk_overlap,
                specific_files=updated_files,
            )
            if chunks:
                rag.add_documents(chunks, doc_metadata, chunk_sources)

        knowledge_base_loaded = True
    else:
        knowledge_base_loaded = False

    return get_document_list(), get_kb_stats()


def process_upload(files):
    """处理文件上传"""
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
        chunks, doc_metadata, chunk_sources = load_documents_from_folder(
            folder_path=str(upload_folder),
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
            specific_files=uploaded_names,
        )
        rag.add_documents(chunks, doc_metadata, chunk_sources)
        msg = f"✅ 已添加: {', '.join(uploaded_names)}"
    else:
        chunks, doc_metadata, chunk_sources = load_documents_from_folder(
            folder_path=str(upload_folder),
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
        )
        rag.build_knowledge_base(chunks, doc_metadata, chunk_sources)
        knowledge_base_loaded = True
        msg = f"✅ 知识库已构建 ({len(chunks)} 个文本块)"

    return get_document_list(), get_kb_stats(), msg


def chat(message, history, conversation_id):
    """处理聊天消息"""
    global rag, knowledge_base_loaded, current_user

    if not knowledge_base_loaded or rag is None:
        # Gradio Chatbot 格式: (user_msg, assistant_msg)
        history.append((message, "⚠️ 请先上传文档构建知识库"))
        return history, conversation_id

    try:
        answer, contexts = rag.answer(
            message,
            use_rerank=config.use_rerank,
            use_query_rewrite=config.use_query_rewrite,
            show_citations=config.show_citations,
        )

        # 保存用户消息到 conversation_manager
        if conversation_id:
            conversation_manager.add_message(
                current_user["username"],
                conversation_id,
                "user",
                message
            )
            conversation_manager.add_message(
                current_user["username"],
                conversation_id,
                "assistant",
                answer,
                {"contexts": contexts} if contexts else None
            )

        # Gradio Chatbot 格式: (user_msg, assistant_msg)
        history.append((message, answer))
        return history, conversation_id

    except Exception as e:
        history.append((message, f"❌ 错误: {str(e)}"))
        return history, conversation_id


def clear_chat():
    """清空当前对话"""
    return [], None


def delete_all_docs():
    """删除所有文档"""
    global rag, knowledge_base_loaded
    if upload_folder.exists():
        supported_exts = {".txt", ".docx", ".xlsx", ".pptx", ".pdf", ".html"}
        for f in upload_folder.iterdir():
            if f.is_file() and f.suffix.lower() in supported_exts:
                f.unlink()
    # 重新初始化 RAGPipeline
    if rag is not None:
        rag = RAGPipeline()
        knowledge_base_loaded = False
    db_path = Path("data/vector_db")
    if db_path.exists():
        shutil.rmtree(db_path)
    return get_document_list(), get_kb_stats(), None, "🗑️ 知识库已清空"


def start_new_conversation():
    """开始新对话"""
    global current_user
    if current_user["username"]:
        conv_id = conversation_manager.create_conversation(current_user["username"])
        return [], conv_id, f"新对话已创建"
    return [], None, "请先登录"


def load_conversation(conversation_id):
    """加载历史对话"""
    global current_user
    if not conversation_id or not current_user["username"]:
        return [], None

    messages = conversation_manager.get_conversation_messages(
        current_user["username"],
        conversation_id
    )

    # Gradio Chatbot 格式: [(user_msg, assistant_msg), ...]
    history = []
    current_pair = [None, None]
    
    for msg in messages:
        if msg["role"] == "user":
            if current_pair[0] is not None and current_pair[1] is not None:
                history.append(tuple(current_pair))
            current_pair = [msg["content"], None]
        elif msg["role"] == "assistant":
            current_pair[1] = msg["content"]
            if current_pair[0] is not None:
                history.append(tuple(current_pair))
                current_pair = [None, None]
    
    if current_pair[0] is not None and current_pair[1] is not None:
        history.append(tuple(current_pair))

    return history, conversation_id


def get_conversation_list():
    """获取对话列表"""
    global current_user
    if not current_user["username"]:
        return []

    conversations = conversation_manager.list_conversations(current_user["username"])

    result = []
    for conv in conversations[:20]:  # 限制显示 20 条
        updated = conv.get("updated_at", "")[:16]  # 取前16个字符（YYYY-MM-DDTHH:MM）
        msg_count = len(conv.get("messages", []))
        result.append(
            f"{conv['title']} | {msg_count}条 | {updated}"
        )

    return result


def delete_conversation(conversation_id):
    """删除对话"""
    global current_user
    if current_user["username"] and conversation_id:
        conversation_manager.delete_conversation(current_user["username"], conversation_id)
    return [], None, "✅ 对话已删除"


# ========== 登录相关函数 ==========

def do_login(username, password):
    """处理登录"""
    global current_user

    if not username or not password:
        return "❌ 请输入用户名和密码", gr.update(), gr.update(), "", ""

    success, session_id, error = auth_manager.login(username, password)

    if success:
        current_user["session_id"] = session_id
        current_user["username"] = username
        valid, _, role = auth_manager.verify_session(session_id)
        current_user["role"] = role

        # 加载知识库
        load_kb()

        return (
            f"✅ 欢迎，{username}！",
            gr.update(visible=True),  # 显示主界面
            gr.update(visible=False),  # 隐藏登录框
            get_document_list(),
            get_kb_stats()
        )
    else:
        return f"❌ {error}", gr.update(), gr.update(), "", ""


def do_register(username, password, confirm_password):
    """处理注册"""
    if not username or not password:
        return "❌ 请输入用户名和密码"

    if len(username) < 3:
        return "❌ 用户名长度至少 3 个字符"

    if len(password) < 6:
        return "❌ 密码长度至少 6 个字符"

    if password != confirm_password:
        return "❌ 两次密码不一致"

    success, error = auth_manager.register_user(username, password)

    if success:
        return f"✅ 注册成功，请登录"
    else:
        return f"❌ {error}"


def do_logout():
    """处理登出"""
    global current_user, rag, knowledge_base_loaded

    if current_user["session_id"]:
        auth_manager.logout(current_user["session_id"])

    current_user = {"session_id": None, "username": None, "role": None}
    knowledge_base_loaded = False

    return (
        gr.update(visible=False),  # 隐藏主界面
        gr.update(visible=True),  # 显示登录框
        "👋 已退出登录",
        [],
        gr.update(value=""),
        "📊 知识库未加载",
        []
    )


# ========== CSS 样式 ==========

CSS = """
/* 整体风格 */
.gradio-container {
    max-width: 100% !important;
}
.login-container {
    max-width: 400px;
    margin: 100px auto;
    padding: 40px;
    background: #f9f9f9;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
}
.sidebar {
    background: #f7f7f8 !important;
    border-right: 1px solid #e5e5e5 !important;
    min-height: 100vh;
    padding: 16px !important;
}
.main-chat {
    background: #ffffff !important;
    min-height: 100vh;
}
.chat-header {
    padding: 12px 16px;
    border-bottom: 1px solid #e5e5e5;
    background: #ffffff;
}
.conversation-item {
    padding: 10px 12px;
    margin: 4px 0;
    border-radius: 8px;
    background: #ffffff;
    border: 1px solid #e5e5e5;
    font-size: 13px;
    cursor: pointer;
    transition: all 0.2s;
}
.conversation-item:hover {
    background: #f0f0f0;
    border-color: #10a37f;
}
.empty-state {
    text-align: center;
    color: #8e8ea0;
    font-size: 13px;
    padding: 24px 0;
}
footer {
    display: none !important;
}
"""


# ========== 界面布局 ==========

with gr.Blocks(title="RAG 知识库问答", css=CSS) as demo:

    # ==================== 登录界面 ====================
    with gr.Group(visible=True) as login_section:
        with gr.Column(elem_classes="login-container"):
            gr.HTML("""
                <div style="text-align:center;margin-bottom:24px;">
                    <h2 style="color:#10a37f;">🤖 RAG 知识库</h2>
                    <p style="color:#666;">智能问答系统</p>
                </div>
            """)

            username_input = gr.Textbox(
                label="用户名",
                placeholder="请输入用户名",
                lines=1,
            )
            password_input = gr.Textbox(
                label="密码",
                placeholder="请输入密码",
                lines=1,
                type="password",
            )
            confirm_password_input = gr.Textbox(
                label="确认密码",
                placeholder="请再次输入密码（注册时）",
                lines=1,
                type="password",
                visible=False,
            )

            login_btn = gr.Button("登录", variant="primary", size="lg")
            register_btn = gr.Button("注册", variant="secondary", size="sm")
            toggle_register_btn = gr.Button("没有账号？注册", variant="stop", size="sm")

            login_msg = gr.Textbox(label=None, interactive=False, show_label=False)

            # 切换登录/注册
            is_registering = gr.State(False)

    # ==================== 主界面 ====================
    with gr.Group(visible=False) as main_interface:
        with gr.Row(equal_height=True):
            # ---------- 左侧边栏 ----------
            with gr.Column(scale=1, elem_classes="sidebar"):
                # 用户信息
                user_info = gr.HTML()

                with gr.Row():
                    new_chat_btn = gr.Button("💬 新对话", variant="primary", size="sm")
                    logout_btn = gr.Button("🚪 退出", variant="stop", size="sm")

                gr.HTML('<div style="margin:16px 0 8px;font-weight:600;color:#666;">历史对话</div>')

                conversation_list = gr.Dropdown(
                    choices=[],
                    label=None,
                    show_label=False,
                    allow_custom_value=True,
                )

                load_conv_btn = gr.Button("📂 加载对话", variant="secondary", size="sm")
                delete_conv_btn = gr.Button("🗑️ 删除对话", variant="stop", size="sm")

                gr.HTML('<div style="margin:16px 0 8px;font-weight:600;color:#666;">📚 知识库文档</div>')

                kb_stats = gr.Textbox(
                    value="📊 加载中...",
                    label=None,
                    interactive=False,
                    container=False,
                )

                doc_list = gr.Textbox(
                    value="暂无文档",
                    label=None,
                    interactive=False,
                    container=False,
                    lines=6,
                )

                upload_btn = gr.UploadButton(
                    "📁 上传文档",
                    file_count="multiple",
                    variant="primary",
                    size="sm",
                )

                delete_btn = gr.Button("🗑️ 清空知识库", variant="stop", size="sm")

                upload_msg = gr.Textbox(
                    label=None,
                    interactive=False,
                    container=False,
                    show_label=False,
                )

            # ---------- 右侧对话区域 ----------
            with gr.Column(scale=3, elem_classes="main-chat"):
                gr.HTML(
                    '<div style="text-align:center;padding:12px 0 8px 0;'
                    'font-size:18px;font-weight:700;color:#10a37f;">'
                    '💬 RAG 知识库问答</div>'
                )

                chatbot = gr.Chatbot(
                    value=[],
                    height=480,
                    show_label=False,
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="输入你的问题，按 Enter 发送...",
                        show_label=False,
                        scale=8,
                        container=False,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1, size="sm")
                    clear_btn = gr.Button("清空", variant="secondary", scale=1, size="sm")

                current_conv_id = gr.State(None)

    # ==================== 事件绑定 ====================

    # 切换登录/注册
    def toggle_register_mode(is_reg):
        return (
            gr.update(visible=not is_reg),  # 登录按钮
            gr.update(visible=is_reg),  # 注册按钮
            gr.update(visible=is_reg),  # 确认密码
            gr.update(value="注册" if is_reg else "登录"),  # 主按钮文本
            gr.update(value="已有账号？登录" if is_reg else "没有账号？注册"),  # 切换按钮文本
            gr.update("" if is_reg else login_msg.value),  # 清除消息
            gr.update(value=is_reg)  # 更新状态
        )

    toggle_register_btn.click(
        toggle_register_mode,
        is_registering,
        [login_btn, register_btn, confirm_password_input, login_btn, toggle_register_btn, login_msg, is_registering]
    )

    def handle_login(username, password):
        return do_login(username, password)

    def handle_register(username, password, confirm_password, is_reg):
        if not is_reg:
            return login_msg.value if login_msg.value else ""
        return do_register(username, password, confirm_password)

    login_btn.click(
        handle_login,
        [username_input, password_input],
        [login_msg, main_interface, login_section, doc_list, kb_stats]
    )

    register_btn.click(
        lambda u, p, cp, ir: handle_register(u, p, cp, ir),
        [username_input, password_input, confirm_password_input, is_registering],
        login_msg
    )

    # 页面加载时更新用户信息和对话列表
    def update_user_info():
        global current_user
        if current_user["username"]:
            role_text = "管理员" if current_user["role"] == "admin" else "用户"
            return f'<div style="padding:8px 0;font-weight:600;color:#10a37f;">👤 {current_user["username"]} ({role_text})</div>'
        return ""

    def load_initial_data():
        doc_list_val, kb_stats_val = load_kb()
        conv_list_val = get_conversation_list()
        return doc_list_val, kb_stats_val, conv_list_val

    demo.load(
        lambda: (update_user_info(), load_initial_data()[0], load_initial_data()[1], load_initial_data()[2]),
        None,
        [user_info, doc_list, kb_stats, conversation_list]
    )

    # 新对话
    new_chat_btn.click(
        start_new_conversation,
        None,
        [chatbot, current_conv_id, upload_msg]
    )

    # 加载对话
    load_conv_btn.click(
        load_conversation,
        conversation_list,
        [chatbot, current_conv_id]
    )

    # 删除对话
    delete_conv_btn.click(
        delete_conversation,
        current_conv_id,
        [chatbot, current_conv_id, upload_msg]
    ).then(
        get_conversation_list,
        None,
        conversation_list
    )

    # 登出
    logout_btn.click(
        do_logout,
        [],
        [main_interface, login_section, login_msg, chatbot, current_conv_id, kb_stats, conversation_list]
    )

    # 发送消息
    send_btn.click(
        chat,
        [msg_input, chatbot, current_conv_id],
        [chatbot, current_conv_id]
    ).then(lambda: "", None, msg_input)

    msg_input.submit(
        chat,
        [msg_input, chatbot, current_conv_id],
        [chatbot, current_conv_id]
    ).then(lambda: "", None, msg_input)

    # 清空对话
    clear_btn.click(
        clear_chat,
        None,
        [chatbot, current_conv_id]
    )

    # 上传文档
    upload_btn.upload(
        process_upload,
        upload_btn,
        [doc_list, kb_stats, upload_msg]
    )

    # 删除知识库
    delete_btn.click(
        delete_all_docs,
        None,
        [doc_list, kb_stats, chatbot, upload_msg]
    )


if __name__ == "__main__":
    print("启动 RAG 知识库问答系统...")
    print("默认管理员账号: admin / admin")
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7860)
