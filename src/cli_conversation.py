"""
命令行会话管理模块

提供命令行界面的会话管理功能：
- 创建、切换、删除会话
- 显示会话列表
- 保存聊天记录

与 ConversationManager 的关系：
- ConversationManager: 负责会话数据的存储和检索（底层）
- CLIConversationManager: 负责命令行交互逻辑（上层）
"""

from typing import Optional, List, Dict


class CLIConversationManager:
    """
    命令行会话管理器

    封装命令行会话管理逻辑，提供简洁的接口。
    """

    def __init__(self, username: str = "cli_user"):
        """
        初始化命令行会话管理器

        Args:
            username: 用户名
        """
        from src.conversation_manager import ConversationManager
        
        self.cm = ConversationManager()
        self.username = username
        self.current_conversation_id: Optional[str] = None
        
        # 自动创建第一个会话
        self.current_conversation_id = self.cm.create_conversation(
            self.username, 
            "新对话"
        )

    def send_message(self, message: str, save: bool = True) -> bool:
        """
        保存用户消息

        Args:
            message: 消息内容
            save: 是否保存

        Returns:
            是否保存成功
        """
        if not save or not self.current_conversation_id:
            return False
        return self.cm.add_message(
            self.username,
            self.current_conversation_id,
            "user",
            message
        )

    def save_response(self, response: str, metadata: dict = None) -> bool:
        """
        保存助手回复

        Args:
            response: 回复内容
            metadata: 额外元数据

        Returns:
            是否保存成功
        """
        if not self.current_conversation_id:
            return False
        return self.cm.add_message(
            self.username,
            self.current_conversation_id,
            "assistant",
            response,
            metadata
        )

    def new_conversation(self, title: str = "新对话") -> str:
        """
        创建新会话

        Args:
            title: 会话标题

        Returns:
            新会话ID
        """
        self.current_conversation_id = self.cm.create_conversation(
            self.username,
            title
        )
        return self.current_conversation_id

    def switch_conversation(self, conversation_id: str) -> tuple[bool, Optional[Dict]]:
        """
        切换到指定会话

        Args:
            conversation_id: 会话ID

        Returns:
            (是否成功, 会话信息)
        """
        conv = self.cm.get_conversation(self.username, conversation_id)
        if conv:
            self.current_conversation_id = conversation_id
            return True, conv
        return False, None

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        删除指定会话

        Args:
            conversation_id: 会话ID

        Returns:
            是否删除成功
        """
        conversations = self.cm._load_conversations(self.username)
        updated = [c for c in conversations if c["id"] != conversation_id]
        
        if len(updated) < len(conversations):
            self.cm._save_conversations(self.username, updated)
            
            # 如果删除的是当前会话，创建新会话
            if self.current_conversation_id == conversation_id:
                self.current_conversation_id = self.cm.create_conversation(
                    self.username,
                    "新对话"
                )
            return True
        return False

    def list_conversations(self, limit: int = 10) -> List[Dict]:
        """
        获取会话列表

        Args:
            limit: 返回数量限制

        Returns:
            会话列表
        """
        return self.cm._load_conversations(self.username)[:limit]

    def get_history(self, limit: int = 10) -> List[Dict]:
        """
        获取当前会话的历史消息

        Args:
            limit: 返回消息数量限制

        Returns:
            消息列表
        """
        if not self.current_conversation_id:
            return []
        
        conv = self.cm.get_conversation(self.username, self.current_conversation_id)
        if not conv:
            return []
        
        messages = conv.get("messages", [])
        return messages[-limit:] if limit else messages

    def get_context(self, max_messages: int = 5) -> List[Dict]:
        """
        获取对话上下文（用于 RAG）

        Args:
            max_messages: 最大消息数

        Returns:
            格式化的上下文消息列表
        """
        if not self.current_conversation_id:
            return []
        
        return self.cm.get_conversation_context(
            self.username,
            self.current_conversation_id,
            max_messages=max_messages
        )


def show_help():
    """显示帮助信息"""
    print("\n" + "=" * 40)
    print("[命令行说明]")
    print("-" * 40)
    print("  直接输入问题进行问答")
    print("  new           - 创建新会话")
    print("  list          - 查看会话列表")
    print("  switch <id>   - 切换到指定会话")
    print("  delete <id>   - 删除指定会话")
    print("  clear         - 清空当前会话")
    print("  history       - 查看当前会话历史")
    print("  help          - 显示帮助信息")
    print("  exit/quit     - 退出程序")
    print("=" * 40)
