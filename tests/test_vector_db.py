"""
测试向量数据库集成（包含自动降级功能）
"""

import sys
import os
from pathlib import Path

# 设置环境变量避免中文输出问题
os.environ['PYTHONIOENCODING'] = 'utf-8'

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

print("测试向量数据库集成...")

try:
    from src.vector_db import VectorDBFactory
    
    print("[OK] 导入成功")
    
    # 测试自动选择模式
    db = VectorDBFactory.create("auto")
    print("[OK] 创建向量数据库实例成功")
    
    # 测试基本操作
    chunks = ["这是第一个测试文档", "这是第二个测试文档", "这是第三个测试文档"]
    vectors = np.random.randn(3, 1536).astype(np.float32)
    
    db.add(chunks, vectors, ["test1.txt", "test2.txt", "test3.txt"])
    print("[OK] 添加文档块成功")
    
    print("文档块数量:", len(db))
    
    # 测试检索
    query_vector = np.random.randn(1, 1536).astype(np.float32)
    results = db.search(query_vector, top_k=2)
    print("[OK] 检索成功，返回", len(results), "条结果")
    
    # 测试混合检索
    results = db.hybrid_search("测试", query_vector, top_k=2)
    print("[OK] 混合检索成功，返回", len(results), "条结果")
    
    # 测试保存和加载
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        db.save(tmpdir)
        print("[OK] 保存成功")
        
        new_db = VectorDBFactory.create("auto")
        new_db.load(tmpdir)
        print("[OK] 加载成功，文档块数量:", len(new_db))
    
    print("\n测试通过！")
    
except Exception as e:
    print("\n测试失败:", str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试工厂模式
print("\n测试 VectorDBFactory...")
try:
    db = VectorDBFactory.create("memory")
    print("[OK] VectorDBFactory.create('memory') 成功")
    
    print("测试通过！")
except Exception as e:
    print("\nVectorDBFactory 测试失败:", str(e))
    sys.exit(1)
