"""
用户管理模块

负责用户的注册、登录、权限管理等功能。
用户数据存储在本地 JSON 文件中。
"""

import json
import os
import hashlib
import secrets
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
from datetime import datetime


@dataclass
class User:
    """用户数据模型"""
    username: str
    password_hash: str
    salt: str
    created_at: str
    role: str = "user"  # admin, user
    last_login: Optional[str] = None
    is_active: bool = True


class UserManager:
    """
    用户管理器

    负责用户的 CRUD 操作，用户数据存储在本地 JSON 文件中。
    默认创建一个 admin 管理员账号。
    """

    def __init__(self, storage_path: str = "data/users.json"):
        """
        初始化用户管理器

        Args:
            storage_path: 用户数据存储路径
        """
        self.storage_path = Path(storage_path)
        self._ensure_storage_dir()
        self._ensure_default_admin()

    def _ensure_storage_dir(self):
        """确保存储目录存在"""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def _ensure_default_admin(self):
        """确保默认管理员账号存在"""
        if not self.storage_path.exists():
            # 创建默认管理员
            self.create_user("admin", "admin", role="admin")
        else:
            # 检查 admin 是否存在
            users = self._load_users()
            if not any(u["username"] == "admin" for u in users):
                self.create_user("admin", "admin", role="admin")

    def _load_users(self) -> List[Dict]:
        """加载用户数据"""
        if not self.storage_path.exists():
            return []
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save_users(self, users: List[Dict]):
        """保存用户数据"""
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=2)

    def _hash_password(self, password: str, salt: str) -> str:
        """密码哈希"""
        return hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        ).hex()

    def _generate_salt(self) -> str:
        """生成盐值"""
        return secrets.token_hex(32)

    def create_user(self, username: str, password: str, role: str = "user") -> bool:
        """
        创建新用户

        Args:
            username: 用户名
            password: 密码
            role: 角色 (admin/user)

        Returns:
            bool: 创建是否成功
        """
        users = self._load_users()

        # 检查用户名是否已存在
        if any(u["username"] == username for u in users):
            return False

        # 生成盐值和密码哈希
        salt = self._generate_salt()
        password_hash = self._hash_password(password, salt)

        # 创建用户
        user = User(
            username=username,
            password_hash=password_hash,
            salt=salt,
            role=role,
            created_at=datetime.now().isoformat(),
            is_active=True
        )

        users.append(asdict(user))
        self._save_users(users)
        return True

    def authenticate(self, username: str, password: str) -> Optional[Dict]:
        """
        用户认证

        Args:
            username: 用户名
            password: 密码

        Returns:
            Optional[Dict]: 认证成功返回用户信息，失败返回 None
        """
        users = self._load_users()

        for user_data in users:
            if user_data["username"] == username and user_data["is_active"]:
                # 验证密码
                password_hash = self._hash_password(password, user_data["salt"])
                if password_hash == user_data["password_hash"]:
                    # 更新最后登录时间
                    user_data["last_login"] = datetime.now().isoformat()
                    self._save_users(users)
                    return user_data

        return None

    def get_user(self, username: str) -> Optional[Dict]:
        """
        获取用户信息

        Args:
            username: 用户名

        Returns:
            Optional[Dict]: 用户信息，不存在返回 None
        """
        users = self._load_users()
        for user_data in users:
            if user_data["username"] == username:
                return user_data
        return None

    def update_password(self, username: str, old_password: str, new_password: str) -> bool:
        """
        更新密码

        Args:
            username: 用户名
            old_password: 旧密码
            new_password: 新密码

        Returns:
            bool: 更新是否成功
        """
        users = self._load_users()

        for i, user_data in enumerate(users):
            if user_data["username"] == username:
                # 验证旧密码
                password_hash = self._hash_password(old_password, user_data["salt"])
                if password_hash != user_data["password_hash"]:
                    return False

                # 更新密码
                new_salt = self._generate_salt()
                users[i]["salt"] = new_salt
                users[i]["password_hash"] = self._hash_password(new_password, new_salt)
                self._save_users(users)
                return True

        return False

    def delete_user(self, username: str) -> bool:
        """
        删除用户

        Args:
            username: 用户名

        Returns:
            bool: 删除是否成功
        """
        users = self._load_users()
        initial_count = len(users)
        users = [u for u in users if u["username"] != username]

        if len(users) < initial_count:
            self._save_users(users)
            return True
        return False

    def list_users(self) -> List[Dict]:
        """
        获取所有用户列表

        Returns:
            List[Dict]: 用户列表
        """
        return self._load_users()

    def change_role(self, username: str, new_role: str) -> bool:
        """
        修改用户角色

        Args:
            username: 用户名
            new_role: 新角色 (admin/user)

        Returns:
            bool: 修改是否成功
        """
        users = self._load_users()

        for i, user_data in enumerate(users):
            if user_data["username"] == username:
                users[i]["role"] = new_role
                self._save_users(users)
                return True

        return False
