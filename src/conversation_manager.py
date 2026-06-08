"""
对话管理模块 - 带长期记忆功能

负责管理用户的对话历史，支持多轮对话上下文和长期记忆。
对话数据存储在本地 JSON 文件中。

长期记忆特性：
1. 对话摘要/总结 - 将长对话总结为简短摘要
2. 记忆向量存储 - 将对话内容向量化存储
3. 记忆检索 - 在回答时检索历史对话中的相关信息
4. 用户画像 - 存储用户的偏好、设定等信息
"""

import json
import uuid
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict
from datetime import datetime

import numpy as np

try:
    from src.embedding import embed_text
    from src.llm import llm_call
    from src.config import config
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False


@dataclass
class Message:
    """消息数据模型"""
    role: str  # user, assistant, system
    content: str
    timestamp: str
    metadata: Optional[Dict] = None


@dataclass
class MemoryItem:
    """记忆项数据模型"""
    id: str
    content: str
    timestamp: str
    conversation_id: str
    embedding: Optional[List[float]] = None  # 向量表示
    tags: List[str] = field(default_factory=list)  # 标签，用于分类


@dataclass
class UserProfile:
    """用户画像数据模型"""
    user_id: str
    name: str = ""
    preferences: Dict = field(default_factory=dict)  # 用户偏好设置
    interests: List[str] = field(default_factory=list)  # 用户兴趣
    settings: Dict = field(default_factory=dict)  # 用户设置
    created_at: str = ""
    updated_at: str = ""


@dataclass
class Conversation:
    """对话数据模型"""
    id: str
    user_id: str
    title: str  # 对话标题，默认取第一条用户消息
    created_at: str
    updated_at: str
    summary: str = ""  # 对话摘要（用于长期记忆）
    messages: List[Dict] = field(default_factory=list)
    is_active: bool = True


class ConversationManager:
    """
    对话管理器（带长期记忆）

    负责对话的创建、查询、删除等操作。
    支持多轮对话上下文管理和长期记忆功能。
    """

    def __init__(self, storage_dir: str = "data/conversations"):
        """
        初始化对话管理器

        Args:
            storage_dir: 对话数据存储目录
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # 记忆向量存储目录
        self.memory_dir = self.storage_dir / "memories"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # 用户画像存储目录
        self.profiles_dir = self.storage_dir / "profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def _get_user_storage_path(self, user_id: str) -> Path:
        """获取用户对话存储文件路径"""
        return self.storage_dir / f"{user_id}_conversations.json"

    def _get_user_memory_path(self, user_id: str) -> Path:
        """获取用户记忆存储文件路径"""
        return self.memory_dir / f"{user_id}_memories.json"

    def _get_user_profile_path(self, user_id: str) -> Path:
        """获取用户画像存储文件路径"""
        return self.profiles_dir / f"{user_id}_profile.json"

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

    def _load_memories(self, user_id: str) -> List[Dict]:
        """加载用户的记忆列表"""
        memory_path = self._get_user_memory_path(user_id)
        if not memory_path.exists():
            return []
        try:
            with open(memory_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save_memories(self, user_id: str, memories: List[Dict]):
        """保存用户记忆列表"""
        memory_path = self._get_user_memory_path(user_id)
        with open(memory_path, 'w', encoding='utf-8') as f:
            json.dump(memories, f, ensure_ascii=False, indent=2)

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

        conversation_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()

        conversation = Conversation(
            id=conversation_id,
            user_id=user_id,
            title=title or "新对话",
            summary="",
            messages=[],
            created_at=now,
            updated_at=now,
            is_active=True
        )

        conversations.insert(0, asdict(conversation))
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

                # 定期更新对话摘要（每5条消息或对话结束时）
                if len(conversations[i]["messages"]) % 5 == 0:
                    self.update_conversation_summary(user_id, conversation_id)

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

        recent_messages = messages[-max_messages:] if max_messages else messages

        context = []
        for msg in recent_messages:
            context.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        return context

    def get_long_term_context(
        self,
        user_id: str,
        query: str,
        max_memories: int = 3
    ) -> List[str]:
        """
        获取长期记忆上下文（从历史对话中检索相关记忆）

        Args:
            user_id: 用户ID
            query: 当前查询
            max_memories: 最大返回记忆数量

        Returns:
            List[str]: 相关记忆内容列表
        """
        if not EMBEDDING_AVAILABLE:
            return []

        memories = self._load_memories(user_id)
        if not memories:
            return []

        try:
            # 获取查询向量
            query_vector = embed_text(query, model=config.embedding_model)
            
            # 计算相似度
            memory_scores = []
            for mem in memories:
                if mem.get("embedding"):
                    embedding = np.array(mem["embedding"])
                    similarity = np.dot(query_vector, embedding) / (np.linalg.norm(query_vector) * np.linalg.norm(embedding))
                    memory_scores.append((mem, float(similarity)))
            
            # 按相似度排序
            memory_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 返回最相关的记忆
            return [mem[0]["content"] for mem in memory_scores[:max_memories]]
        
        except Exception as e:
            print(f"[长期记忆] 检索失败: {e}")
            return []

    def update_conversation_summary(self, user_id: str, conversation_id: str) -> bool:
        """
        更新对话摘要（使用LLM生成）

        Args:
            user_id: 用户ID
            conversation_id: 对话ID

        Returns:
            bool: 更新是否成功
        """
        if not EMBEDDING_AVAILABLE:
            return False

        conversation = self.get_conversation(user_id, conversation_id)
        if not conversation:
            return False

        messages = conversation.get("messages", [])
        if len(messages) < 2:
            return False

        try:
            # 构建总结提示词
            conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            
            prompt = f"""请为以下对话生成一个简洁的摘要：

对话内容：
{conversation_text}

摘要要求：
1. 不超过50字
2. 包含关键要点
3. 使用中文

请直接输出摘要，不要输出其他内容。
"""

            result = llm_call(
                model=config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                api_key=config.dashscope_api_key,
                base_url=config.llm_base_url,
            )

            # 更新对话摘要
            conversations = self._load_conversations(user_id)
            for i, conv in enumerate(conversations):
                if conv["id"] == conversation_id:
                    conversations[i]["summary"] = result.strip()
                    self._save_conversations(user_id, conversations)
                    return True

        except Exception as e:
            print(f"[对话摘要] 生成失败: {e}")
            return False

        return False

    def add_memory(
        self,
        user_id: str,
        content: str,
        conversation_id: str = "",
        tags: Optional[List[str]] = None
    ) -> str:
        """
        添加长期记忆

        Args:
            user_id: 用户ID
            content: 记忆内容
            conversation_id: 关联的对话ID
            tags: 标签列表

        Returns:
            str: 记忆ID
        """
        memories = self._load_memories(user_id)

        memory_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()

        # 生成记忆向量（如果可用）
        embedding = None
        if EMBEDDING_AVAILABLE:
            try:
                embedding = embed_text(content, model=config.embedding_model).tolist()
            except Exception as e:
                print(f"[记忆向量] 生成失败: {e}")

        memory = MemoryItem(
            id=memory_id,
            content=content,
            timestamp=now,
            conversation_id=conversation_id,
            embedding=embedding,
            tags=tags or []
        )

        memories.append(asdict(memory))
        self._save_memories(user_id, memories)
        return memory_id

    def extract_and_save_memories(self, user_id: str, conversation_id: str) -> bool:
        """
        从对话中提取重要信息并保存为长期记忆

        Args:
            user_id: 用户ID
            conversation_id: 对话ID

        Returns:
            bool: 是否成功
        """
        if not EMBEDDING_AVAILABLE:
            return False

        conversation = self.get_conversation(user_id, conversation_id)
        if not conversation:
            return False

        messages = conversation.get("messages", [])
        if not messages:
            return False

        try:
            # 构建提取提示词
            conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            
            prompt = f"""请从以下对话中提取重要信息作为长期记忆：

对话内容：
{conversation_text}

提取规则：
1. 只提取事实性信息（姓名、日期、数字、偏好、重要事件等）
2. 每条记忆简洁明了，不超过30字
3. 不要提取问题，只提取答案和事实
4. 输出格式：每行一条记忆

请直接输出记忆内容，不要输出其他内容。
"""

            result = llm_call(
                model=config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                api_key=config.dashscope_api_key,
                base_url=config.llm_base_url,
            )

            # 解析并保存记忆
            memory_lines = [line.strip() for line in result.strip().split("\n") if line.strip()]
            for line in memory_lines:
                self.add_memory(user_id, line, conversation_id)

            print(f"[长期记忆] 从对话中提取了 {len(memory_lines)} 条记忆")
            return True

        except Exception as e:
            print(f"[长期记忆] 提取失败: {e}")
            return False

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

    def get_user_profile(self, user_id: str) -> Dict:
        """
        获取用户画像

        Args:
            user_id: 用户ID

        Returns:
            Dict: 用户画像信息
        """
        profile_path = self._get_user_profile_path(user_id)
        if not profile_path.exists():
            # 创建默认画像
            profile = UserProfile(
                user_id=user_id,
                name="",
                preferences={},
                interests=[],
                settings={},
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(profile), f, ensure_ascii=False, indent=2)
            return asdict(profile)

        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def update_user_profile(self, user_id: str, updates: Dict) -> bool:
        """
        更新用户画像

        Args:
            user_id: 用户ID
            updates: 更新的字段

        Returns:
            bool: 更新是否成功
        """
        profile = self.get_user_profile(user_id)
        if not profile:
            return False

        profile.update(updates)
        profile["updated_at"] = datetime.now().isoformat()

        profile_path = self._get_user_profile_path(user_id)
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
        return True

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

        for i, conv in enumerate(conversations):
            if conv["id"] == conversation_id:
                del conversations[i]
                self._save_conversations(user_id, conversations)
                return True

        return False
