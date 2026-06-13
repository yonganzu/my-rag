"""
RAG 知识库问答系统 - 现代化界面
"""

import os
import sys
sys.path.insert(0, 'E:\\something\\大模型上课\\learn\\local_packages')
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
        return f"📊 {doc_count} 文档 · {chunk_count} 文本块"
    return "📊 知识库未加载"


def load_kb():
    """加载知识库"""
    global rag, knowledge_base_loaded
    if rag is None:
        rag = RAGPipeline()
        rag.set_conversation_manager(conversation_manager)

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
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "⚠️ 请先上传文档构建知识库"})
        return history, conversation_id

    try:
        # 如果没有对话ID，自动创建新对话
        if not conversation_id and current_user["username"]:
            conversation_id = conversation_manager.create_conversation(current_user["username"])

        long_term_memories = []
        if current_user["username"]:
            long_term_memories = conversation_manager.get_long_term_context(
                current_user["username"],
                message,
                max_memories=3
            )

        if conversation_id and current_user["username"]:
            conversation_manager.add_message(
                current_user["username"],
                conversation_id,
                "user",
                message
            )

        user_role = current_user.get("role") or "user"
        answer, contexts = rag.answer(
            message,
            use_rerank=config.use_rerank,
            use_query_rewrite=config.use_query_rewrite,
            show_citations=config.show_citations,
            user_role=user_role,
            user_id=current_user["username"],
            conversation_id=conversation_id or "",
        )

        if long_term_memories:
            memory_text = "\n".join([f"- {mem}" for mem in long_term_memories])
            answer = f"📖 根据历史对话，我记得：\n{memory_text}\n\n{answer}"

        if conversation_id and current_user["username"]:
            conversation_manager.add_message(
                current_user["username"],
                conversation_id,
                "assistant",
                answer,
                {"contexts": contexts} if contexts else None
            )

            conv = conversation_manager.get_conversation(current_user["username"], conversation_id)
            if conv and len(conv.get("messages", [])) >= 5:
                conversation_manager.extract_and_save_memories(current_user["username"], conversation_id)

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})
        return history, conversation_id

    except Exception as e:
        error_msg = f"❌ 错误: {str(e)}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
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
    if rag is not None:
        rag = RAGPipeline()
        rag.set_conversation_manager(conversation_manager)
        knowledge_base_loaded = False
    db_path = Path("data/vector_db")
    if db_path.exists():
        shutil.rmtree(db_path)
    return get_document_list(), get_kb_stats(), "", "🗑️ 知识库已清空"


def start_new_conversation():
    """开始新对话"""
    global current_user
    if current_user["username"]:
        conv_id = conversation_manager.create_conversation(current_user["username"])
        conv_list = get_conversation_list()
        return [], conv_id, f"新对话已创建", gr.update(choices=conv_list)
    return [], None, "请先登录", gr.update(choices=[])


def load_conversation(conv_id):
    """加载历史对话"""
    global current_user
    if not conv_id or not current_user["username"]:
        return [], None

    # 解析下拉框格式: conv_id|display_text
    if '|' in conv_id:
        conv_id = conv_id.split('|')[0]

    messages = conversation_manager.get_conversation_messages(
        current_user["username"],
        conv_id
    )

    # Gradio 6.x Chatbot 使用字典格式: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    history = []
    
    for msg in messages:
        history.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    return history, conv_id


def get_conversation_list():
    """获取对话列表（下拉选项格式）"""
    global current_user
    if not current_user["username"]:
        return []

    conversations = conversation_manager.list_conversations(current_user["username"])

    if not conversations:
        return []

    choices = []
    for conv in conversations[:20]:
        msg_count = len(conv.get("messages", []))
        # 只显示有消息的对话
        if msg_count == 0:
            continue
        updated = conv.get("updated_at", "")[:16]
        title = conv['title']
        # 使用 conv_id 作为值，显示文本作为标签
        display_text = f"📝 {title} ({msg_count}条) · {updated}"
        # Gradio 6.x Dropdown 使用列表格式
        choices.append(f"{conv['id']}|{display_text}")

    return choices


def delete_conversation(conv_id):
    """删除对话"""
    global current_user
    if current_user["username"] and conv_id:
        # 解析下拉框格式: conv_id|display_text
        if '|' in conv_id:
            conv_id = conv_id.split('|')[0]
        conversation_manager.delete_conversation(current_user["username"], conv_id)
    conv_list = get_conversation_list()
    return [], None, "✅ 对话已删除", gr.update(choices=conv_list)


# ==================== 上下文管理函数 ====================

def trim_context_handler(conv_id):
    """修剪上下文（保留最近20条消息）"""
    global current_user
    if not conv_id or not current_user["username"]:
        return "请先选择对话", gr.update(visible=False)
    
    # 解析对话ID
    if '|' in conv_id:
        conv_id = conv_id.split('|')[0]
    
    success = conversation_manager.trim_context(current_user["username"], conv_id, max_messages=20)
    if success:
        return "✅ 已修剪上下文，保留最近20条消息", gr.update(visible=False)
    else:
        return "❌ 修剪失败", gr.update(visible=False)


def compress_context_handler(conv_id):
    """压缩上下文（将早期消息合并为摘要）"""
    global current_user
    if not conv_id or not current_user["username"]:
        return "请先选择对话", gr.update(visible=False)
    
    # 解析对话ID
    if '|' in conv_id:
        conv_id = conv_id.split('|')[0]
    
    success = conversation_manager.compress_context(current_user["username"], conv_id)
    if success:
        return "✅ 已压缩上下文，早期消息已合并为摘要", gr.update(visible=False)
    else:
        return "❌ 压缩失败（需要LLM支持）", gr.update(visible=False)


def clear_context_handler(conv_id):
    """清空上下文"""
    global current_user
    if not conv_id or not current_user["username"]:
        return [], None, "请先选择对话", gr.update(visible=False)
    
    # 解析对话ID
    if '|' in conv_id:
        conv_id = conv_id.split('|')[0]
    
    success = conversation_manager.clear_context(current_user["username"], conv_id)
    if success:
        return [], None, "✅ 已清空对话上下文", gr.update(visible=False)
    else:
        return [], None, "❌ 清空失败", gr.update(visible=False)


def get_context_stats_handler(conv_id):
    """获取上下文统计信息"""
    global current_user
    if not conv_id or not current_user["username"]:
        return "请先选择对话", gr.update(visible=True)
    
    # 解析对话ID
    if '|' in conv_id:
        conv_id = conv_id.split('|')[0]
    
    stats = conversation_manager.get_context_stats(current_user["username"], conv_id)
    if stats:
        stats_text = f"""📊 对话统计：
标题: {stats['title']}
消息总数: {stats['total_messages']} (用户:{stats['user_messages']}, 助手:{stats['assistant_messages']})
字符总数: {stats['total_characters']}
平均长度: {stats['avg_message_length']}字"""
        return stats_text, gr.update(visible=True)
    else:
        return "❌ 获取统计失败", gr.update(visible=True)


def do_login(username, password):
    """处理登录"""
    global current_user

    if not username or not password:
        return "❌ 请输入用户名和密码", gr.update(), gr.update(), "", "", gr.update(choices=[])

    success, session_id, error = auth_manager.login(username, password)

    if success:
        current_user["session_id"] = session_id
        current_user["username"] = username
        valid, _, role = auth_manager.verify_session(session_id)
        current_user["role"] = role
        load_kb()

        conv_list = get_conversation_list()
        
        return (
            f"✅ 欢迎，{username}！",
            gr.update(visible=True),
            gr.update(visible=False),
            get_document_list(),
            get_kb_stats(),
            gr.update(choices=conv_list)
        )
    else:
        return f"❌ {error}", gr.update(), gr.update(), "", "", gr.update(choices=[])


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
        gr.update(visible=False),
        gr.update(visible=True),
        "👋 已退出登录",
        gr.update(choices=[]),
        gr.update(value=""),
        "📊 知识库未加载",
        gr.update(choices=[])
    )


# ========== CSS 样式 ==========
CSS = """
/* 整体风格 */
:root {
    --primary-color: #6366f1;
    --primary-dark: #4f46e5;
    --secondary-color: #8b5cf6;
    --success-color: #10b981;
    --warning-color: #f59e0b;
    --danger-color: #ef4444;
    --bg-primary: #ffffff;
    --bg-secondary: #f8fafc;
    --bg-tertiary: #f1f5f9;
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --text-muted: #94a3b8;
    --border-color: #e2e8f0;
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
}

.gradio-container {
    max-width: 100% !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
}

/* 登录界面 */
.login-wrapper {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.login-card {
    background: white;
    border-radius: 20px;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    padding: 48px;
    width: 100%;
    max-width: 420px;
}

.login-header {
    text-align: center;
    margin-bottom: 32px;
}

.login-title {
    font-size: 28px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 8px;
}

.login-subtitle {
    color: var(--text-secondary);
    font-size: 14px;
}

.login-form {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

/* 按钮样式 */
button {
    font-weight: 500 !important;
    border-radius: var(--radius-md) !important;
    transition: all 0.2s ease !important;
}

button:hover:not(:disabled) {
    transform: translateY(-1px);
}

button:active:not(:disabled) {
    transform: translateY(0);
}

.btn-primary {
    background: var(--primary-color) !important;
    border: none !important;
    color: white !important;
    height: 48px !important;
    font-size: 16px !important;
}

.btn-primary:hover:not(:disabled) {
    background: var(--primary-dark) !important;
    box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.4);
}

.btn-secondary {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    height: 40px !important;
}

.btn-secondary:hover:not(:disabled) {
    background: var(--bg-secondary) !important;
}

.btn-danger {
    background: transparent !important;
    border: 1px solid var(--danger-color) !important;
    color: var(--danger-color) !important;
    height: 40px !important;
}

.btn-danger:hover:not(:disabled) {
    background: rgba(239, 68, 68, 0.1) !important;
}

/* 输入框样式 */
input, textarea {
    border-radius: var(--radius-md) !important;
    border: 1.5px solid var(--border-color) !important;
    padding: 14px 16px !important;
    font-size: 15px !important;
    transition: all 0.2s ease !important;
}

input:focus, textarea:focus {
    outline: none !important;
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
}

/* 侧边栏 */
.sidebar {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-color) !important;
    min-height: 100vh;
    padding: 20px !important;
    width: 280px !important;
    flex-shrink: 0;
}

/* 用户信息 */
.user-info {
    padding: 16px;
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    border-radius: var(--radius-lg);
    color: white;
    margin-bottom: 20px;
}

.user-name {
    font-size: 16px;
    font-weight: 600;
}

.user-role {
    font-size: 12px;
    opacity: 0.9;
    margin-top: 4px;
}

/* 会话列表 */
.conv-list {
    background: white;
    border-radius: var(--radius-md);
    border: 1px solid var(--border-color);
    overflow: hidden;
    max-height: 240px;
    overflow-y: auto;
}

.conv-item {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-color);
    cursor: pointer;
    transition: background 0.2s;
}

.conv-item:last-child {
    border-bottom: none;
}

.conv-item:hover {
    background: var(--bg-secondary);
}

.conv-item.active {
    background: rgba(99, 102, 241, 0.1);
    border-left: 3px solid var(--primary-color);
}

.conv-title {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.conv-meta {
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 4px;
}

.conv-list {
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    padding: 8px;
    background: var(--bg-primary);
    max-height: 200px;
    overflow-y: auto;
}

.conv-list li {
    padding: 10px 12px;
    margin-bottom: 4px;
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: all 0.2s ease;
    font-size: 13px;
    color: var(--text-primary);
    border: 1px solid transparent;
}

.conv-list li:hover {
    background: var(--bg-secondary);
    border-color: var(--border-color);
}

.conv-list li:last-child {
    margin-bottom: 0;
}

.conv-item-button {
    width: 100%;
    text-align: left;
    padding: 10px 12px;
    margin-bottom: 4px;
    border-radius: 8px;
    border: 1px solid transparent;
    background: transparent;
    cursor: pointer;
    transition: all 0.2s ease;
    font-family: inherit;
}

.conv-item-button:hover {
    background: var(--bg-secondary);
    border-color: var(--border-color);
}

.conv-item-button:active {
    background: rgba(99, 102, 241, 0.1);
    border-color: var(--primary-color);
}

/* 聊天区域 */
.chat-container {
    background: var(--bg-primary);
    display: flex;
    flex-direction: column;
    height: 100vh;
}

.chat-header {
    padding: 16px 24px;
    border-bottom: 1px solid var(--border-color);
    background: white;
}

.chat-title {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
}

.chat-subtitle {
    font-size: 13px;
    color: var(--text-muted);
    margin-top: 4px;
}

/* 聊天消息 */
.chatbot {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.message-wrapper {
    display: flex;
    gap: 12px;
}

.message-wrapper.user {
    justify-content: flex-end;
}

.message-wrapper.assistant {
    justify-content: flex-start;
}

.message-avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
}

.message-avatar.user {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    color: white;
}

.message-avatar.assistant {
    background: var(--bg-tertiary);
    color: var(--text-secondary);
}

.message-content {
    max-width: 70%;
    padding: 12px 16px;
    border-radius: var(--radius-md);
    font-size: 15px;
    line-height: 1.6;
}

.message-content.user {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    color: white;
    border-radius: var(--radius-md) var(--radius-md) 4px var(--radius-md);
}

.message-content.assistant {
    background: var(--bg-tertiary);
    color: var(--text-primary);
    border-radius: var(--radius-md) var(--radius-md) var(--radius-md) 4px;
}

/* 输入区域 */
.input-area {
    padding: 16px 24px;
    border-top: 1px solid var(--border-color);
    background: white;
}

.input-row {
    display: flex;
    gap: 12px;
}

.input-field {
    flex: 1;
    border-radius: var(--radius-md) !important;
}

/* 知识库面板 */
.kb-panel {
    background: white;
    border-radius: var(--radius-md);
    border: 1px solid var(--border-color);
    padding: 16px;
    margin-top: 16px;
}

.kb-title {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.kb-stats {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 12px;
    padding: 8px 12px;
    background: var(--bg-secondary);
    border-radius: var(--radius-sm);
}

.doc-list {
    font-size: 13px;
    color: var(--text-secondary);
    white-space: pre-wrap;
    max-height: 120px;
    overflow-y: auto;
}

/* 滚动条样式 */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: var(--text-muted);
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--text-secondary);
}

/* 隐藏默认 footer */
footer {
    display: none !important;
}

/* 状态消息 */
.status-msg {
    padding: 12px 16px;
    border-radius: var(--radius-md);
    font-size: 14px;
    margin-top: 16px;
    text-align: center;
}

.status-success {
    background: rgba(16, 185, 129, 0.1);
    color: var(--success-color);
}

.status-error {
    background: rgba(239, 68, 68, 0.1);
    color: var(--danger-color);
}
"""

# ========== 空的 JavaScript 代码（保留占位符）==========
JS = ""


# ========== 界面布局 ==========
with gr.Blocks(title="RAG 知识库问答") as demo:

    # ==================== 登录界面 ====================
    with gr.Group(visible=True) as login_section:
        gr.HTML("""
            <div class="login-wrapper">
                <div class="login-card">
                    <div class="login-header">
                        <div style="font-size: 48px; margin-bottom: 16px;">🤖</div>
                        <h2 class="login-title">RAG 知识库问答</h2>
                        <p class="login-subtitle">基于大语言模型的智能问答系统</p>
                    </div>
                    <div class="login-form">
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
            placeholder="请再次输入密码",
            lines=1,
            type="password",
            visible=False,
        )

        with gr.Row():
            login_btn = gr.Button("登录", variant="primary", size="lg", elem_classes="btn-primary")
            register_btn = gr.Button("注册", variant="secondary", size="sm", visible=False, elem_classes="btn-secondary")
        
        toggle_register_btn = gr.Button("没有账号？立即注册", variant="stop", size="sm")
        login_msg = gr.Textbox(label=None, interactive=False, show_label=False, container=False)

        gr.HTML("""
                    </div>
                </div>
            </div>
        """)

        is_registering = gr.State(False)

    # ==================== 主界面 ====================
    with gr.Group(visible=False) as main_interface:
        with gr.Row(equal_height=True):
            # ---------- 左侧边栏 ----------
            with gr.Column(elem_classes="sidebar"):
                # 用户信息卡片
                user_info = gr.HTML("""
                    <div class="user-info">
                        <div class="user-name">用户名</div>
                        <div class="user-role">用户</div>
                    </div>
                """)

                # 操作按钮
                with gr.Row():
                    new_chat_btn = gr.Button("💬 新对话", variant="primary", size="sm", elem_classes="btn-primary")
                    logout_btn = gr.Button("🚪 退出", variant="stop", size="sm", elem_classes="btn-danger")

                # 会话列表
                gr.HTML('<div class="kb-title">📝 历史对话</div>')
                conv_list_container = gr.Group(elem_classes="conv-list-container")
                with conv_list_container:
                    conversation_list = gr.Dropdown(
                        choices=[],
                        label=None,
                        show_label=False,
                        multiselect=False,
                        container=False,
                        elem_classes="conv-dropdown"
                    )
                
                conversation_id_display = gr.Textbox(
                    label=None,
                    show_label=False,
                    container=False,
                    visible=False,
                    value=""
                )

                with gr.Row():
                    load_conv_btn = gr.Button("📂 加载", variant="secondary", size="sm", elem_classes="btn-secondary")
                    delete_conv_btn = gr.Button("🗑️ 删除", variant="stop", size="sm", elem_classes="btn-danger")

                # 上下文管理面板
                with gr.Group(elem_classes="context-panel"):
                    gr.HTML('<div class="kb-title">⚙️ 上下文管理</div>')
                    with gr.Row():
                        trim_btn = gr.Button("✂️ 修剪", variant="secondary", size="sm", elem_classes="btn-secondary")
                        compress_btn = gr.Button("📦 压缩", variant="secondary", size="sm", elem_classes="btn-secondary")
                    with gr.Row():
                        clear_btn = gr.Button("🧹 清空", variant="stop", size="sm", elem_classes="btn-danger")
                        stats_btn = gr.Button("📊 统计", variant="secondary", size="sm", elem_classes="btn-secondary")
                    context_stats_output = gr.Textbox(
                        label=None,
                        interactive=False,
                        container=False,
                        show_label=False,
                        lines=3,
                        elem_classes="context-stats",
                        visible=False
                    )

                # 知识库面板
                with gr.Group(elem_classes="kb-panel"):
                    gr.HTML('<div class="kb-title">📚 知识库文档</div>')
                    kb_stats = gr.Textbox(
                        value="📊 加载中...",
                        label=None,
                        interactive=False,
                        container=False,
                        elem_classes="kb-stats"
                    )
                    doc_list = gr.Textbox(
                        value="暂无文档",
                        label=None,
                        interactive=False,
                        container=False,
                        lines=5,
                        elem_classes="doc-list"
                    )

                # 上传区域
                upload_btn = gr.UploadButton(
                    "📁 上传文档",
                    file_count="multiple",
                    variant="primary",
                    size="sm",
                    elem_classes="btn-primary"
                )
                delete_btn = gr.Button("🗑️ 清空知识库", variant="stop", size="sm", elem_classes="btn-danger")
                upload_msg = gr.Textbox(
                    label=None,
                    interactive=False,
                    container=False,
                    show_label=False,
                    lines=1,
                    visible=False
                )

            # ---------- 右侧对话区域 ----------
            with gr.Column(elem_classes="chat-container"):
                # 聊天头部
                with gr.Row(elem_classes="chat-header"):
                    with gr.Column():
                        gr.HTML('<div class="chat-title">💬 RAG 知识库问答</div>')
                        gr.HTML('<div class="chat-subtitle">基于文档的智能问答系统</div>')

                # 聊天消息区域
                chatbot = gr.Chatbot(
                    value=[],
                    show_label=False,
                )

                # 输入区域
                with gr.Row(elem_classes="input-area"):
                    msg_input = gr.Textbox(
                        placeholder="输入你的问题，按 Enter 发送...",
                        show_label=False,
                        container=False,
                        elem_classes="input-field",
                        autofocus=True
                    )
                    send_btn = gr.Button("发送", variant="primary", size="sm", elem_classes="btn-primary")
                    clear_btn = gr.Button("清空", variant="secondary", size="sm", elem_classes="btn-secondary")

                current_conv_id = gr.State(None)

    # ==================== 事件绑定 ====================

    # 切换登录/注册
    def toggle_register_mode(is_reg):
        return (
            gr.update(visible=not is_reg),
            gr.update(visible=is_reg),
            gr.update(visible=is_reg),
            gr.update(value="注册" if is_reg else "登录"),
            gr.update(value="已有账号？登录" if is_reg else "没有账号？立即注册"),
            gr.update(""),
            gr.update(value=is_reg)
        )

    toggle_register_btn.click(
        toggle_register_mode,
        inputs=[is_registering],
        outputs=[login_btn, register_btn, confirm_password_input, login_btn, toggle_register_btn, login_msg, is_registering]
    )

    # 登录
    login_btn.click(
        do_login,
        inputs=[username_input, password_input],
        outputs=[login_msg, main_interface, login_section, doc_list, kb_stats, conversation_list]
    )

    # 注册
    register_btn.click(
        do_register,
        inputs=[username_input, password_input, confirm_password_input],
        outputs=[login_msg]
    )

    # 登出
    logout_btn.click(
        do_logout,
        outputs=[main_interface, login_section, login_msg, conversation_list, doc_list, kb_stats, conversation_list]
    )

    # 上传文档
    upload_btn.upload(
        process_upload,
        inputs=[upload_btn],
        outputs=[doc_list, kb_stats, upload_msg]
    )

    # 清空知识库
    delete_btn.click(
        delete_all_docs,
        outputs=[doc_list, kb_stats, upload_msg, login_msg]
    )

    # 发送消息
    send_btn.click(
        chat,
        inputs=[msg_input, chatbot, current_conv_id],
        outputs=[chatbot, current_conv_id]
    )
    msg_input.submit(
        chat,
        inputs=[msg_input, chatbot, current_conv_id],
        outputs=[chatbot, current_conv_id]
    )

    # 清空聊天
    clear_btn.click(
        clear_chat,
        outputs=[chatbot, current_conv_id]
    )

    # 新对话
    new_chat_btn.click(
        start_new_conversation,
        outputs=[chatbot, current_conv_id, login_msg, conversation_list]
    )

    # 下拉框选择变化时自动加载对话
    conversation_list.change(
        load_conversation,
        inputs=[conversation_list],
        outputs=[chatbot, current_conv_id]
    )

    # 加载对话按钮
    load_conv_btn.click(
        load_conversation,
        inputs=[conversation_list],
        outputs=[chatbot, current_conv_id]
    )

    # 删除对话
    delete_conv_btn.click(
        delete_conversation,
        inputs=[conversation_list],
        outputs=[chatbot, current_conv_id, login_msg, conversation_list]
    )

    # 上下文管理 - 修剪上下文
    trim_btn.click(
        trim_context_handler,
        inputs=[conversation_list],
        outputs=[login_msg, context_stats_output]
    )

    # 上下文管理 - 压缩上下文
    compress_btn.click(
        compress_context_handler,
        inputs=[conversation_list],
        outputs=[login_msg, context_stats_output]
    )

    # 上下文管理 - 清空上下文
    clear_btn.click(
        clear_context_handler,
        inputs=[conversation_list],
        outputs=[chatbot, current_conv_id, login_msg, context_stats_output]
    )

    # 上下文管理 - 获取统计
    stats_btn.click(
        get_context_stats_handler,
        inputs=[conversation_list],
        outputs=[context_stats_output, context_stats_output]
    )


if __name__ == "__main__":
    # 添加自定义 JavaScript
    demo.load(js=JS)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        debug=True
    )
