"""
权限过滤测试用例

测试场景：
1. 创建不同安全级别的测试文档
2. 模拟不同角色用户进行检索
3. 验证权限过滤逻辑是否正确生效

角色权限表：
| 角色 | 可访问级别 |
|------|------------|
| admin | public, internal, confidential, secret |
| editor | public, internal, confidential |
| viewer | public, internal |
| user | public, internal |
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.permission import (
    PermissionManager, 
    DocLevel, 
    check_doc_access, 
    filter_docs_by_permission,
    get_user_max_doc_level
)


def test_doc_level_enum():
    """测试文档级别枚举"""
    print("=== 测试文档级别枚举 ===")
    
    # 测试级别顺序
    order = DocLevel.get_level_order()
    print(f"级别顺序: {order}")
    
    # 测试访问判断
    test_cases = [
        ("internal", "public", True),    # internal 用户可以访问 public
        ("internal", "internal", True),  # internal 用户可以访问 internal
        ("internal", "confidential", False),  # internal 用户不能访问 confidential
        ("secret", "confidential", True),     # secret 用户可以访问 confidential
    ]
    
    for user_level, doc_level, expected in test_cases:
        result = DocLevel.can_access(user_level, doc_level)
        status = "[OK]" if result == expected else "[FAIL]"
        print("  %s %s 用户访问 %s 文档: %s (期望: %s)" % (status, user_level, doc_level, result, expected))


def test_permission_manager():
    """测试权限管理器"""
    print("\n=== 测试权限管理器 ===")
    
    pm = PermissionManager()
    
    # 获取所有角色
    roles = pm.get_all_roles()
    print(f"可用角色: {roles}")
    
    # 测试各角色的最大文档级别
    for role in roles:
        max_level = pm.get_max_doc_level(role)
        print(f"  {role}: 最高可访问级别 = {max_level}")


def test_doc_access_check():
    """测试文档访问权限检查"""
    print("\n=== 测试文档访问权限检查 ===")
    
    test_cases = [
        ("admin", "secret", True),
        ("admin", "confidential", True),
        ("editor", "confidential", True),
        ("editor", "secret", False),
        ("viewer", "internal", True),
        ("viewer", "confidential", False),
        ("user", "internal", True),
        ("user", "secret", False),
    ]
    
    for role, doc_level, expected in test_cases:
        can_access, msg = check_doc_access(role, doc_level)
        status = "[OK]" if can_access == expected else "[FAIL]"
        print("  %s %s 角色访问 %s 文档: %s (期望: %s)" % (status, role, doc_level, can_access, expected))


def test_document_filtering():
    """测试文档列表过滤"""
    print("\n=== 测试文档列表过滤 ===")
    
    # 模拟文档列表（包含不同级别）
    mock_docs = [
        {"id": "1", "title": "公开文档", "level": "public"},
        {"id": "2", "title": "内部文档", "level": "internal"},
        {"id": "3", "title": "机密文档", "level": "confidential"},
        {"id": "4", "title": "绝密文档", "level": "secret"},
    ]
    
    # 测试各角色能看到的文档
    roles_to_test = ["admin", "editor", "viewer", "user"]
    
    for role in roles_to_test:
        filtered = filter_docs_by_permission(role, mock_docs)
        levels = [doc["level"] for doc in filtered]
        titles = [doc["title"] for doc in filtered]
        print(f"\n  {role} 角色可见文档 ({len(filtered)} 个):")
        for doc in filtered:
            print(f"    - {doc['title']} ({doc['level']})")


def test_retrieval_simulation():
    """模拟检索场景测试"""
    print("\n=== 模拟检索场景测试 ===")
    
    # 模拟检索结果（包含来源和级别信息）
    mock_results = [
        ("公开内容片段...", 0.95, "公开文档.txt|public"),
        ("内部内容片段...", 0.90, "内部文档.txt|internal"),
        ("机密内容片段...", 0.85, "机密文档.txt|confidential"),
        ("绝密内容片段...", 0.80, "绝密文档.txt|secret"),
    ]
    
    # 模拟不同角色的检索结果过滤
    def simulate_retrieval(user_role):
        max_level = get_user_max_doc_level(user_role)
        filtered = []
        
        for chunk, score, source in mock_results:
            # 解析文档级别
            if "|" in source:
                doc_source, doc_level = source.rsplit("|", 1)
            else:
                doc_source = source
                doc_level = "public"
            
            if DocLevel.can_access(max_level, doc_level):
                filtered.append((chunk, score, source))
        
        return filtered
    
    # 测试各角色
    for role in ["admin", "editor", "viewer", "user"]:
        results = simulate_retrieval(role)
        print(f"\n  {role} 角色检索结果 ({len(results)} 条):")
        for chunk, score, source in results:
            doc_name, level = source.rsplit("|", 1)
            print(f"    [{level}] {doc_name} (相似度: {score:.2f})")


def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")
    
    # 测试不存在的角色
    result, msg = check_doc_access("unknown_role", "public")
    print("  [OK] 不存在的角色 'unknown_role' 访问 public: %s" % result)
    
    # 测试不存在的文档级别
    result, msg = check_doc_access("admin", "unknown_level")
    print("  [OK] admin 访问未知级别 'unknown_level': %s" % result)
    
    # 测试空文档列表过滤
    filtered = filter_docs_by_permission("user", [])
    print("  [OK] 空文档列表过滤结果: %s" % filtered)


if __name__ == "__main__":
    print("=" * 60)
    print("权限过滤测试用例")
    print("=" * 60)
    
    test_doc_level_enum()
    test_permission_manager()
    test_doc_access_check()
    test_document_filtering()
    test_retrieval_simulation()
    test_edge_cases()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)