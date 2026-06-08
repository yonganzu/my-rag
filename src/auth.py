"""
认证模块

提供基于 Session 的用户认证功能。
支持登录、登出、会话验证等操作。
"""

import secrets
import time
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class Session:
    """会话数据模型"""
    session_id: str
    username: str
    role: str
    created_at: float  # Unix 时间戳
    expires_at: float  # Unix 时间戳


class SessionManager:
    """
    会话管理器

    负责管理用户登录会话，使用内存存储。
    """

    def __init__(self, session_timeout: int = 24 * 60 * 60):  # 默认 24 小时
        """
        初始化会话管理器

        Args:
            session_timeout: 会话超时时间（秒），默认 24 小时
        """
        self.sessions: Dict[str, Session] = {}
        self.session_timeout = session_timeout

    def _generate_session_id(self) -> str:
        """生成安全的会话 ID"""
        return secrets.token_urlsafe(32)

    def create_session(self, username: str, role: str) -> str:
        """
        创建新会话

        Args:
            username: 用户名
            role: 用户角色

        Returns:
            str: 会话 ID
        """
        session_id = self._generate_session_id()
        now = time.time()

        session = Session(
            session_id=session_id,
            username=username,
            role=role,
            created_at=now,
            expires_at=now + self.session_timeout
        )

        self.sessions[session_id] = session
        return session_id

    def get_session(self, session_id: str) -> Optional[Session]:
        """
        获取会话信息

        Args:
            session_id: 会话 ID

        Returns:
            Optional[Session]: 会话信息，不存在或已过期返回 None
        """
        session = self.sessions.get(session_id)
        if not session:
            return None

        # 检查是否过期
        if time.time() > session.expires_at:
            self.delete_session(session_id)
            return None

        return session

    def refresh_session(self, session_id: str) -> bool:
        """
        刷新会话有效期

        Args:
            session_id: 会话 ID

        Returns:
            bool: 刷新是否成功
        """
        session = self.sessions.get(session_id)
        if not session:
            return False

        # 重置过期时间
        session.expires_at = time.time() + self.session_timeout
        return True

    def delete_session(self, session_id: str) -> bool:
        """
        删除会话

        Args:
            session_id: 会话 ID

        Returns:
            bool: 删除是否成功
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def cleanup_expired_sessions(self):
        """清理所有过期的会话"""
        now = time.time()
        expired = [
            sid for sid, session in self.sessions.items()
            if now > session.expires_at
        ]
        for sid in expired:
            del self.sessions[sid]

    def get_user_sessions(self, username: str) -> list:
        """
        获取用户的所有会话

        Args:
            username: 用户名

        Returns:
            list: 会话列表
        """
        return [
            {
                "session_id": s.session_id,
                "created_at": datetime.fromtimestamp(s.created_at).isoformat(),
                "expires_at": datetime.fromtimestamp(s.expires_at).isoformat(),
                "is_active": time.time() <= s.expires_at
            }
            for s in self.sessions.values()
            if s.username == username
        ]

    def logout_user(self, username: str) -> int:
        """
        登出用户（删除该用户所有会话）

        Args:
            username: 用户名

        Returns:
            int: 删除的会话数量
        """
        sessions_to_delete = [
            sid for sid, s in self.sessions.items()
            if s.username == username
        ]
        for sid in sessions_to_delete:
            del self.sessions[sid]
        return len(sessions_to_delete)


class AuthManager:
    """
    认证管理器

    整合用户管理和会话管理，提供统一的认证接口。
    """

    def __init__(self):
        """初始化认证管理器"""
        self.user_manager = None  # 延迟初始化
        self.session_manager = SessionManager()

    def _get_user_manager(self):
        """延迟加载用户管理器"""
        if self.user_manager is None:
            from src.user_manager import UserManager
            self.user_manager = UserManager()
        return self.user_manager

    def login(self, username: str, password: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        用户登录

        Args:
            username: 用户名
            password: 密码

        Returns:
            Tuple[bool, Optional[str], Optional[str]]:
            (是否成功, session_id或None, 错误信息或None)
        """
        user_manager = self._get_user_manager()
        user = user_manager.authenticate(username, password)

        if not user:
            return False, None, "用户名或密码错误"

        # 创建会话
        session_id = self.session_manager.create_session(
            username=user["username"],
            role=user["role"]
        )

        return True, session_id, None

    def logout(self, session_id: str) -> bool:
        """
        用户登出

        Args:
            session_id: 会话 ID

        Returns:
            bool: 登出是否成功
        """
        return self.session_manager.delete_session(session_id)

    def verify_session(self, session_id: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        验证会话

        Args:
            session_id: 会话 ID

        Returns:
            Tuple[bool, Optional[str], Optional[str]]:
            (是否有效, 用户名或None, 角色或None)
        """
        session = self.session_manager.get_session(session_id)
        if not session:
            return False, None, None

        # 刷新会话有效期
        self.session_manager.refresh_session(session_id)
        return True, session.username, session.role

    def is_admin(self, session_id: str) -> bool:
        """
        检查是否为管理员

        Args:
            session_id: 会话 ID

        Returns:
            bool: 是否为管理员
        """
        valid, _, role = self.verify_session(session_id)
        return valid and role == "admin"

    def register_user(self, username: str, password: str) -> Tuple[bool, Optional[str]]:
        """
        注册新用户

        Args:
            username: 用户名
            password: 密码

        Returns:
            Tuple[bool, Optional[str]]: (是否成功, 错误信息或None)
        """
        user_manager = self._get_user_manager()

        if len(username) < 3:
            return False, "用户名长度至少 3 个字符"

        if len(password) < 6:
            return False, "密码长度至少 6 个字符"

        success = user_manager.create_user(username, password)
        if not success:
            return False, "用户名已存在"

        return True, None

    def change_password(
        self,
        session_id: str,
        old_password: str,
        new_password: str
    ) -> Tuple[bool, Optional[str]]:
        """
        修改密码

        Args:
            session_id: 会话 ID
            old_password: 旧密码
            new_password: 新密码

        Returns:
            Tuple[bool, Optional[str]]: (是否成功, 错误信息或None)
        """
        valid, username, _ = self.verify_session(session_id)
        if not valid:
            return False, "会话无效，请重新登录"

        if len(new_password) < 6:
            return False, "新密码长度至少 6 个字符"

        user_manager = self._get_user_manager()
        success = user_manager.update_password(username, old_password, new_password)

        if not success:
            return False, "旧密码不正确"

        return True, None


# 全局认证管理器实例
auth_manager = AuthManager()
