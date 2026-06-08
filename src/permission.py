"""
权限管理模块

实现基于角色的访问控制（RBAC）+ 文档级别控制：
1. 权限定义 - 定义系统中所有可控制的权限
2. 角色定义 - 将权限组合成角色
3. 文档级别 - 文档的安全级别（public/internal/confidential/secret）
4. 权限检查 - 检查用户是否拥有特定权限或能否访问特定级别的文档
5. 装饰器 - 简化权限检查的使用

设计原则：
- 解耦：权限检查作为装饰器，不侵入业务逻辑
- 简洁：权限配置用 JSON 文件存储，无需数据库
- 灵活：支持动态添加角色和权限
- 硬控制：文档有明确的访问级别，检索时强制过滤
"""

import json
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable, Set
from functools import wraps


class Permission(Enum):
    """
    权限定义
    
    命名规范：资源:操作
    - doc: 文档相关权限
    - user: 用户管理权限
    - conv: 对话相关权限
    - system: 系统配置权限
    """
    # 文档权限
    DOC_READ = "doc:read"           # 查看文档
    DOC_WRITE = "doc:write"         # 上传/修改文档
    DOC_DELETE = "doc:delete"       # 删除文档
    DOC_MANAGE = "doc:manage"       # 管理所有文档（包括其他用户的）
    
    # 用户权限
    USER_READ = "user:read"         # 查看用户信息
    USER_CREATE = "user:create"     # 创建用户
    USER_UPDATE = "user:update"     # 修改用户信息
    USER_DELETE = "user:delete"     # 删除用户
    USER_MANAGE = "user:manage"     # 管理所有用户
    
    # 对话权限
    CONV_READ = "conv:read"         # 查看对话
    CONV_CREATE = "conv:create"     # 创建对话
    CONV_DELETE = "conv:delete"     # 删除对话
    CONV_MANAGE = "conv:manage"     # 管理所有对话（包括其他用户的）
    
    # 系统权限
    SYSTEM_CONFIG = "system:config" # 系统配置
    SYSTEM_VIEW_LOGS = "system:logs"  # 查看系统日志


class DocLevel(Enum):
    """
    文档安全级别
    
    级别从低到高：public < internal < confidential < secret
    用户只能访问其角色允许的级别及以下的文档
    """
    PUBLIC = "public"           # 公开：所有人可访问
    INTERNAL = "internal"       # 内部：登录用户可访问
    CONFIDENTIAL = "confidential"  # 机密：编辑者及以上可访问
    SECRET = "secret"           # 绝密：仅管理员可访问
    
    @classmethod
    def get_level_order(cls) -> Dict[str, int]:
        """获取级别顺序映射"""
        return {
            cls.PUBLIC.value: 0,
            cls.INTERNAL.value: 1,
            cls.CONFIDENTIAL.value: 2,
            cls.SECRET.value: 3,
        }
    
    @classmethod
    def can_access(cls, user_max_level: str, doc_level: str) -> bool:
        """
        检查用户能否访问某级别的文档
        
        Args:
            user_max_level: 用户可访问的最高级别
            doc_level: 文档的级别
            
        Returns:
            是否可以访问
        """
        order = cls.get_level_order()
        user_order = order.get(user_max_level, 0)
        doc_order = order.get(doc_level, 0)
        return user_order >= doc_order


@dataclass
class Role:
    """
    角色定义
    
    Attributes:
        name: 角色名称（唯一标识）
        description: 角色描述
        permissions: 该角色拥有的权限列表
        inherits: 继承的其他角色
        max_doc_level: 可访问的最高文档级别
    """
    name: str
    description: str = ""
    permissions: Set[str] = field(default_factory=set)
    inherits: List[str] = field(default_factory=list)
    max_doc_level: str = DocLevel.PUBLIC.value  # 默认只能访问公开文档
    
    def has_permission(self, permission: str) -> bool:
        """检查是否拥有某个权限"""
        return permission in self.permissions


class PermissionManager:
    """
    权限管理器
    
    负责角色的定义、存储和权限检查。
    支持文档级别的访问控制。
    """
    
    # 默认角色配置（包含文档级别）
    DEFAULT_ROLES = {
        "admin": {
            "description": "管理员，拥有所有权限，可访问所有级别文档",
            "permissions": [p.value for p in Permission],
            "inherits": [],
            "max_doc_level": DocLevel.SECRET.value  # 可访问绝密文档
        },
        "editor": {
            "description": "编辑者，可以管理文档和对话，可访问机密文档",
            "permissions": [
                Permission.DOC_READ.value,
                Permission.DOC_WRITE.value,
                Permission.DOC_DELETE.value,
                Permission.CONV_READ.value,
                Permission.CONV_CREATE.value,
                Permission.CONV_DELETE.value,
            ],
            "inherits": [],
            "max_doc_level": DocLevel.CONFIDENTIAL.value  # 可访问机密文档
        },
        "viewer": {
            "description": "查看者，只能查看内容，可访问内部文档",
            "permissions": [
                Permission.DOC_READ.value,
                Permission.CONV_READ.value,
            ],
            "inherits": [],
            "max_doc_level": DocLevel.INTERNAL.value  # 可访问内部文档
        },
        "user": {
            "description": "普通用户，可以问答和查看文档，可访问公开文档",
            "permissions": [
                Permission.DOC_READ.value,
                Permission.CONV_READ.value,
                Permission.CONV_CREATE.value,
                Permission.CONV_DELETE.value,
            ],
            "inherits": [],
            "max_doc_level": DocLevel.INTERNAL.value  # 可访问内部文档
        }
    }
    
    def __init__(self, config_path: str = "data/roles.json"):
        """
        初始化权限管理器
        
        Args:
            config_path: 角色配置文件路径
        """
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.roles: Dict[str, Role] = {}
        self._load_roles()
    
    def _load_roles(self):
        """加载角色配置"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    roles_data = json.load(f)
                for name, data in roles_data.items():
                    self.roles[name] = Role(
                        name=name,
                        description=data.get("description", ""),
                        permissions=set(data.get("permissions", [])),
                        inherits=data.get("inherits", []),
                        max_doc_level=data.get("max_doc_level", DocLevel.PUBLIC.value)
                    )
                return
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[权限] 加载配置失败: {e}，使用默认配置")
        
        # 使用默认配置
        self._init_default_roles()
    
    def _init_default_roles(self):
        """初始化默认角色"""
        for name, data in self.DEFAULT_ROLES.items():
            self.roles[name] = Role(
                name=name,
                description=data["description"],
                permissions=set(data["permissions"]),
                inherits=data.get("inherits", []),
                max_doc_level=data.get("max_doc_level", DocLevel.PUBLIC.value)
            )
        self._save_roles()
    
    def _save_roles(self):
        """保存角色配置"""
        roles_data = {}
        for name, role in self.roles.items():
            roles_data[name] = {
                "description": role.description,
                "permissions": list(role.permissions),
                "inherits": role.inherits,
                "max_doc_level": role.max_doc_level
            }
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(roles_data, f, ensure_ascii=False, indent=2)
    
    def get_role(self, role_name: str) -> Optional[Role]:
        """
        获取角色
        
        Args:
            role_name: 角色名称
            
        Returns:
            Role 对象，不存在返回 None
        """
        return self.roles.get(role_name)
    
    def get_all_roles(self) -> List[str]:
        """获取所有角色名称"""
        return list(self.roles.keys())
    
    def create_role(
        self,
        name: str,
        permissions: List[str],
        description: str = "",
        inherits: List[str] = None,
        max_doc_level: str = DocLevel.PUBLIC.value
    ) -> bool:
        """
        创建新角色
        
        Args:
            name: 角色名称
            permissions: 权限列表
            description: 角色描述
            inherits: 继承的角色
            max_doc_level: 可访问的最高文档级别
            
        Returns:
            是否创建成功
        """
        if name in self.roles:
            return False
        
        self.roles[name] = Role(
            name=name,
            description=description,
            permissions=set(permissions),
            inherits=inherits or [],
            max_doc_level=max_doc_level
        )
        self._save_roles()
        return True
    
    def update_role(
        self,
        name: str,
        permissions: List[str] = None,
        description: str = None,
        inherits: List[str] = None,
        max_doc_level: str = None
    ) -> bool:
        """
        更新角色
        
        Args:
            name: 角色名称
            permissions: 新权限列表
            description: 新描述
            inherits: 新继承角色
            max_doc_level: 新文档级别
            
        Returns:
            是否更新成功
        """
        if name not in self.roles:
            return False
        
        role = self.roles[name]
        if permissions is not None:
            role.permissions = set(permissions)
        if description is not None:
            role.description = description
        if inherits is not None:
            role.inherits = inherits
        if max_doc_level is not None:
            role.max_doc_level = max_doc_level
        
        self._save_roles()
        return True
    
    def delete_role(self, name: str) -> bool:
        """
        删除角色
        
        Args:
            name: 角色名称
            
        Returns:
            是否删除成功
        """
        if name not in self.roles:
            return False
        
        # 不允许删除默认角色
        if name in self.DEFAULT_ROLES:
            return False
        
        del self.roles[name]
        self._save_roles()
        return True
    
    def get_effective_permissions(self, role_name: str) -> Set[str]:
        """
        获取角色的有效权限（包括继承的权限）
        
        Args:
            role_name: 角色名称
            
        Returns:
            有效权限集合
        """
        role = self.roles.get(role_name)
        if not role:
            return set()
        
        # 获取直接权限
        permissions = set(role.permissions)
        
        # 获取继承的权限
        for inherited_role in role.inherits:
            permissions.update(self.get_effective_permissions(inherited_role))
        
        return permissions
    
    def get_max_doc_level(self, role_name: str) -> str:
        """
        获取角色可访问的最高文档级别
        
        Args:
            role_name: 角色名称
            
        Returns:
            最高文档级别
        """
        role = self.roles.get(role_name)
        if not role:
            return DocLevel.PUBLIC.value
        return role.max_doc_level
    
    def has_permission(self, role_name: str, permission: str) -> bool:
        """
        检查角色是否拥有某个权限
        
        Args:
            role_name: 角色名称
            permission: 权限标识
            
        Returns:
            是否拥有该权限
        """
        effective_permissions = self.get_effective_permissions(role_name)
        return permission in effective_permissions
    
    def can_access_doc(self, role_name: str, doc_level: str) -> bool:
        """
        检查角色能否访问某级别的文档
        
        Args:
            role_name: 角色名称
            doc_level: 文档级别
            
        Returns:
            是否可以访问
        """
        max_level = self.get_max_doc_level(role_name)
        return DocLevel.can_access(max_level, doc_level)
    
    def filter_docs_by_level(
        self,
        role_name: str,
        docs: List[Dict],
        level_field: str = "level"
    ) -> List[Dict]:
        """
        根据角色过滤文档列表
        
        Args:
            role_name: 角色名称
            docs: 文档列表
            level_field: 文档中级别字段的名称
            
        Returns:
            过滤后的文档列表
        """
        max_level = self.get_max_doc_level(role_name)
        return [
            doc for doc in docs
            if DocLevel.can_access(max_level, doc.get(level_field, DocLevel.PUBLIC.value))
        ]
    
    def check_permission(self, role_name: str, permission: str) -> tuple[bool, str]:
        """
        检查权限并返回结果和消息
        
        Args:
            role_name: 角色名称
            permission: 权限标识
            
        Returns:
            (是否通过, 错误消息)
        """
        if self.has_permission(role_name, permission):
            return True, ""
        return False, f"权限不足：需要 {permission} 权限"
    
    def check_doc_access(self, role_name: str, doc_level: str) -> tuple[bool, str]:
        """
        检查文档访问权限
        
        Args:
            role_name: 角色名称
            doc_level: 文档级别
            
        Returns:
            (是否通过, 错误消息)
        """
        if self.can_access_doc(role_name, doc_level):
            return True, ""
        return False, f"权限不足：无法访问 {doc_level} 级别的文档"


# 全局权限管理器实例
_permission_manager: Optional[PermissionManager] = None


def get_permission_manager() -> PermissionManager:
    """获取全局权限管理器实例"""
    global _permission_manager
    if _permission_manager is None:
        _permission_manager = PermissionManager()
    return _permission_manager


def require_permission(permission: str):
    """
    权限检查装饰器
    
    用法：
        @require_permission("doc:delete")
        def delete_document(doc_id):
            ...
    
    Args:
        permission: 需要的权限标识
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 从 kwargs 中获取 user_role 参数
            user_role = kwargs.get("user_role", "user")
            
            pm = get_permission_manager()
            has_perm, msg = pm.check_permission(user_role, permission)
            
            if not has_perm:
                raise PermissionError(msg)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def check_user_permission(user_role: str, permission: str) -> tuple[bool, str]:
    """
    检查用户权限的便捷函数
    
    Args:
        user_role: 用户角色
        permission: 权限标识
        
    Returns:
        (是否通过, 错误消息)
    """
    pm = get_permission_manager()
    return pm.check_permission(user_role, permission)


def check_doc_access(user_role: str, doc_level: str) -> tuple[bool, str]:
    """
    检查文档访问权限的便捷函数
    
    Args:
        user_role: 用户角色
        doc_level: 文档级别
        
    Returns:
        (是否通过, 错误消息)
    """
    pm = get_permission_manager()
    return pm.check_doc_access(user_role, doc_level)


def filter_docs_by_permission(user_role: str, docs: List[Dict], level_field: str = "level") -> List[Dict]:
    """
    根据用户角色过滤文档列表的便捷函数
    
    Args:
        user_role: 用户角色
        docs: 文档列表
        level_field: 文档中级别字段的名称
        
    Returns:
        过滤后的文档列表
    """
    pm = get_permission_manager()
    return pm.filter_docs_by_level(user_role, docs, level_field)


def get_user_max_doc_level(user_role: str) -> str:
    """
    获取用户可访问的最高文档级别
    
    Args:
        user_role: 用户角色
        
    Returns:
        最高文档级别
    """
    pm = get_permission_manager()
    return pm.get_max_doc_level(user_role)
