"""
对话管理模块

负责管理用户的对话历史，支持多轮对话上下文。
对话数据存储在本地 JSON 文件中。
"""

import json
import uuid
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict
from datetime import datetime


@dataclass
class Message:
    """消息数据模型"""
    role: str  # user, assistant, system
    content: str
    timestamp: str
    metadata: Optional[Dict] = None


@dataclass
class Conversation:
    """对话数据模型"""
    id: str
    user_id: str
    title: str  # 对话标题，默认取第一条用户消息
    created_at: str
    updated_at: str
    messages: List[Dict] = field(default_factory=list)
    is_active: bool = True


class ConversationManager:
    """
    对话管理器

    负责对话的创建、查询、删除等操作。
    支持多轮对话上下文管理。
    """

    def __init__(self, storage_dir: str = "data/conversations"):
        """
        初始化对话管理器

        Args:
            storage_dir: 对话数据存储目录
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_user_storage_path(self, user_id: str) -> Path:
        """获取用户对话存储文件路径"""
        return self.storage_dir / f"{user_id}_conversations.json"

    def _load_conversations(self, user_id: str) -> List[Dict]:
        """加载用户的对话列表"""
        storage_path = self._get_user_storage_path(user_id)
        if not storage_path.exists():
            return []
        try:
            with open(storage_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save_conversations(self, user_id: str, conversations: List[Dict]):
        """保存用户对话列表"""
        storage_path = self._get_user_storage_path(user_id)
        with open(storage_path, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)

    def create_conversation(self, user_id: str, title: Optional[str] = None) -> str:
        """
        创建新对话

        Args:
            user_id: 用户ID
            title: 对话标题，不提供则使用"新对话"

        Returns:
            str: 对话ID
        """
        conversations = self._load_conversations(user_id)

        # 生成唯一ID
        conversation_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()

        conversation = Conversation(
            id=conversation_id,
            user_id=user_id,
            title=title or "新对话",
            messages=[],
            created_at=now,
            updated_at=now,
            is_active=True
        )

        conversations.insert(0, asdict(conversation))  # 新对话放在最前面
        self._save_conversations(user_id, conversations)
        return conversation_id

    def add_message(
        self,
        user_id: str,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        添加消息到对话

        Args:
            user_id: 用户ID
            conversation_id: 对话ID
            role: 角色 (user/assistant/system)
            content: 消息内容
            metadata: 额外元数据（如引用来源）

        Returns:
            bool: 添加是否成功
        """
        conversations = self._load_conversations(user_id)

        for i, conv in enumerate(conversations):
            if conv["id"] == conversation_id:
                message = Message(
                    role=role,
                    content=content,
                    timestamp=datetime.now().isoformat(),
                    metadata=metadata
                )
                conversations[i]["messages"].append(asdict(message))
                conversations[i]["updated_at"] = datetime.now().isoformat()

                # 如果是第一条用户消息，更新对话标题
                if role == "user" and len(conversations[i]["messages"]) == 1:
                    conversations[i]["title"] = content[:30] + ("..." if len(content) > 30 else "")

                self._save_conversations(user_id, conversations)
                return True

        return False

    def get_conversation(self, user_id: str, conversation_id: str) -> Optional[Dict]:
        """
        获取对话详情

        Args:
            user_id: 用户ID
            conversation_id: 对话ID

        Returns:
            Optional[Dict]: 对话信息，不存在返回 None
        """
        conversations = self._load_conversations(user_id)
        for conv in conversations:
            if conv["id"] == conversation_id:
                return conv
        return None

    def get_conversation_messages(
        self,
        user_id: str,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        获取对话消息

        Args:
            user_id: 用户ID
            conversation_id: 对话ID
            limit: 限制返回的消息数量（从最新开始）

        Returns:
            List[Dict]: 消息列表
        """
        conversation = self.get_conversation(user_id, conversation_id)
        if not conversation:
            return []

        messages = conversation.get("messages", [])
        if limit:
            messages = messages[-limit:]
        return messages

    def get_conversation_context(
        self,
        user_id: str,
        conversation_id: str,
        max_messages: int = 10
    ) -> List[Dict]:
        """
        获取对话上下文（用于 RAG）

        Args:
            user_id: 用户ID
            conversation_id: 对话ID
            max_messages: 最大返回消息数

        Returns:
            List[Dict]: 格式化的上下文消息列表
        """
        messages = self.get_conversation_messages(user_id, conversation_id)
        if not messages:
            return []

        # 取最近的消息
        recent_messages = messages[-max_messages:] if max_messages else messages

        # 格式化为对话上下文
        context = []
        for msg in recent_messages:
            context.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        return context

    def list_conversations(self, user_id: str, limit: int = 50) -> List[Dict]:
        """
        获取用户的对话列表

        Args:
            user_id: 用户ID
            limit: 限制返回数量

        Returns:
            List[Dict]: 对话列表（按更新时间倒序）
        """
        conversations = self._load_conversations(user_id)
        return conversations[:limit]

    def update_conversation_title(self, user_id: str, conversation_id: str, title: str) -> bool:
        """
        更新对话标题

        Args:
            user_id: 用户ID
            conversation_id: 对话ID
            title: 新标题

        Returns:
            bool: 更新是否成功
        """
        conversations = self._load_conversations(user_id)

        for i, conv in enumerate(conversations):
            if conv["id"] == conversation_id:
                conversations[i]["title"] = title
                conversations[i]["updated_at"] = datetime.now().isoformat()
                self._save_conversations(user_id, conversations)
                return True

        return False

    def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        """
        删除对话

        Args:
            user_id: 用户ID
            conversation_id: 对话ID

        Returns:
            bool: 删除是否成功
        """
        conversations = self._load_conversations(user_id)
        initial_count = len(conversations)
        conversations = [c for c in conversations if c["id"] != conversation_id]

        if len(conversations) < initial_count:
            self._save_conversations(user_id, conversations)
            return True
        return False

    def search_conversations(self, user_id: str, keyword: str) -> List[Dict]:
        """
        搜索对话

        Args:
            user_id: 用户ID
            keyword: 搜索关键词

        Returns:
            List[Dict]: 匹配的对话列表
        """
        conversations = self._load_conversations(user_id)
        keyword = keyword.lower()
        results = []

        for conv in conversations:
            # 搜索标题
            if keyword in conv["title"].lower():
                results.append(conv)
                continue

            # 搜索消息内容
            for msg in conv.get("messages", []):
                if keyword in msg["content"].lower():
                    results.append(conv)
                    break

        return results

    def get_statistics(self, user_id: str) -> Dict:
        """
        获取用户对话统计

        Args:
            user_id: 用户ID

        Returns:
            Dict: 统计数据
        """
        conversations = self._load_conversations(user_id)
        total_messages = sum(len(c.get("messages", [])) for c in conversations)

        return {
            "total_conversations": len(conversations),
            "total_messages": total_messages,
            "active_conversations": sum(1 for c in conversations if c.get("is_active", True))
        }
