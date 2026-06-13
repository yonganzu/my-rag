"""
会话存储和记忆测试用例

测试场景：
1. 创建新会话
2. 添加消息（用户消息 + 助手回复）
3. 保存会话
4. 加载会话（验证消息配对正确性）
5. 测试会话列表管理
6. 测试会话记忆（上下文管理）
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 创建临时目录用于测试
test_data_dir = project_root / "data" / "test_conversations"
test_data_dir.mkdir(parents=True, exist_ok=True)

# 设置测试数据目录
os.environ["CONVERSATION_DIR"] = str(test_data_dir)

from src.conversation_manager import ConversationManager


def test_create_conversation():
    """测试创建新会话"""
    print("=== 测试创建新会话 ===")
    
    cm = ConversationManager(str(test_data_dir))
    conv_id = cm.create_conversation("test_user", "测试会话")
    
    print("  创建会话ID: %s" % conv_id)
    
    # 检查会话是否存在
    conv = cm.get_conversation("test_user", conv_id)
    assert conv is not None, "会话创建失败"
    assert conv["id"] == conv_id, "会话ID不匹配"
    assert conv["title"] == "测试会话", "会话标题不匹配"
    assert "messages" in conv, "缺少消息列表"
    
    print("  [OK] 会话创建成功")
    return conv_id


def test_add_and_save_messages():
    """测试添加消息和保存会话"""
    print("\n=== 测试添加消息和保存会话 ===")
    
    cm = ConversationManager(str(test_data_dir))
    
    # 创建会话
    conv_id = cm.create_conversation("test_user", "消息测试会话")
    
    # 添加用户消息
    user_msg_id = cm.add_message("test_user", conv_id, "user", "你好，这是第一个问题")
    print("  添加用户消息ID: %s" % user_msg_id)
    
    # 添加助手回复
    assistant_msg_id = cm.add_message("test_user", conv_id, "assistant", "您好！我是智能助手，有什么可以帮助您的？")
    print("  添加助手消息ID: %s" % assistant_msg_id)
    
    # 添加第二条用户消息
    user_msg_id2 = cm.add_message("test_user", conv_id, "user", "什么是RAG？")
    print("  添加第二条用户消息ID: %s" % user_msg_id2)
    
    # 添加第二条助手回复
    assistant_msg_id2 = cm.add_message("test_user", conv_id, "assistant", "RAG是Retrieval-Augmented Generation的缩写，即检索增强生成。")
    print("  添加第二条助手消息ID: %s" % assistant_msg_id2)
    
    # add_message 已自动保存会话
    print("  [OK] 会话保存成功")
    
    return conv_id


def test_load_conversation():
    """测试加载会话（验证消息配对）"""
    print("\n=== 测试加载会话 ===")
    
    cm = ConversationManager(str(test_data_dir))
    
    # 创建并保存会话
    conv_id = cm.create_conversation("test_user", "加载测试会话")
    
    # 添加多条消息（模拟真实对话）
    cm.add_message("test_user", conv_id, "user", "你好")
    cm.add_message("test_user", conv_id, "assistant", "您好！")
    cm.add_message("test_user", conv_id, "user", "今天天气怎么样？")
    cm.add_message("test_user", conv_id, "assistant", "今天天气晴朗，温度25度。")
    cm.add_message("test_user", conv_id, "user", "谢谢")
    cm.add_message("test_user", conv_id, "assistant", "不客气！")
    
    # add_message 已自动保存会话
    
    # 创建新的管理器实例（模拟重启）
    cm_new = ConversationManager(str(test_data_dir))
    
    # 加载会话
    conv = cm_new.get_conversation("test_user", conv_id)
    
    # 验证消息数量
    messages = conv.get("messages", [])
    print("  加载的消息数量: %d" % len(messages))
    
    # 验证消息配对
    for i, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content", "")[:20] + "..." if len(msg.get("content", "")) > 20 else msg.get("content", "")
        print("  消息%d: role=%s, content=%s" % (i+1, role, content))
    
    # 验证消息顺序和配对
    assert len(messages) == 6, "消息数量不匹配"
    assert messages[0]["role"] == "user", "第一条消息应该是用户消息"
    assert messages[1]["role"] == "assistant", "第二条消息应该是助手消息"
    assert messages[2]["role"] == "user", "第三条消息应该是用户消息"
    assert messages[3]["role"] == "assistant", "第四条消息应该是助手消息"
    
    print("  [OK] 会话加载成功，消息配对正确")


def test_conversation_list():
    """测试会话列表管理"""
    print("\n=== 测试会话列表管理 ===")
    
    cm = ConversationManager(str(test_data_dir))
    
    # 使用内部方法获取会话列表
    initial_convs = cm._load_conversations("test_user")
    initial_count = len(initial_convs)
    print("  初始会话数量: %d" % initial_count)
    
    # 创建多个会话
    conv_ids = []
    for i in range(3):
        conv_id = cm.create_conversation("test_user", "会话%d" % (i+1))
        conv_ids.append(conv_id)
        cm.add_message("test_user", conv_id, "user", "测试消息%d" % (i+1))
        cm.add_message("test_user", conv_id, "assistant", "回复%d" % (i+1))
    
    # 获取更新后的会话列表
    conv_list = cm._load_conversations("test_user")
    print("  创建后会话数量: %d" % len(conv_list))
    
    # 验证会话列表
    assert len(conv_list) == initial_count + 3, "会话列表数量不正确"
    
    # 打印会话列表
    print("  会话列表:")
    for conv in conv_list[:5]:  # 只显示前5个
        print("    - %s: %s" % (conv["id"], conv["title"]))
    
    print("  [OK] 会话列表管理正常")


def test_conversation_memory():
    """测试会话记忆（上下文管理）"""
    print("\n=== 测试会话记忆 ===")
    
    cm = ConversationManager(str(test_data_dir))
    
    # 创建会话
    conv_id = cm.create_conversation("test_user", "记忆测试会话")
    
    # 添加上下文消息
    cm.add_message("test_user", conv_id, "user", "我叫张三")
    cm.add_message("test_user", conv_id, "assistant", "您好张三！很高兴认识您。")
    cm.add_message("test_user", conv_id, "user", "我喜欢编程")
    cm.add_message("test_user", conv_id, "assistant", "编程是一项很棒的技能！")
    
    # add_message 已自动保存会话
    
    # 获取会话历史作为上下文（返回 List[Dict]）
    context = cm.get_conversation_context("test_user", conv_id)
    print("  上下文消息数量: %d" % len(context))
    
    # 将上下文转换为字符串以便检查
    context_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context])
    print("  上下文内容预览: %s..." % context_str[:100])
    
    # 验证上下文包含所有消息
    assert "张三" in context_str, "上下文缺少用户名称"
    assert "编程" in context_str, "上下文缺少用户兴趣"
    
    # 测试获取最近消息
    recent = cm.get_conversation_messages("test_user", conv_id, limit=2)
    print("  最近2条消息数量: %d" % len(recent))
    assert len(recent) == 2, "获取最近消息数量不正确"
    
    print("  [OK] 会话记忆功能正常")


def test_empty_conversation():
    """测试空会话处理"""
    print("\n=== 测试空会话处理 ===")
    
    cm = ConversationManager(str(test_data_dir))
    
    # 获取不存在的会话
    conv = cm.get_conversation("test_user", "non_existent_id")
    assert conv is None, "不存在的会话应该返回None"
    print("  [OK] 不存在的会话返回None")
    
    # 获取空会话列表
    # 清理测试目录
    import shutil
    for f in test_data_dir.glob("*.json"):
        f.unlink()
    
    cm_new = ConversationManager(str(test_data_dir))
    conv_list = cm_new._load_conversations("test_user")
    assert len(conv_list) == 0, "空目录应该返回空列表"
    print("  [OK] 空目录返回空会话列表")


if __name__ == "__main__":
    print("=" * 60)
    print("会话存储和记忆测试")
    print("=" * 60)
    
    try:
        test_create_conversation()
        test_add_and_save_messages()
        test_load_conversation()
        test_conversation_list()
        test_conversation_memory()
        test_empty_conversation()
        
        print("\n" + "=" * 60)
        print("所有测试通过！")
        print("=" * 60)
    finally:
        # 清理测试数据
        import shutil
        if test_data_dir.exists():
            shutil.rmtree(test_data_dir)
            print("\n  已清理测试数据")