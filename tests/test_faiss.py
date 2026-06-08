"""
测试 FAISS 向量数据库集成
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("测试 FAISS 向量数据库集成...")

try:
    # 测试导入
    from src.vector_db import VectorDBFactory, FAISSVectorDB, MemoryVectorDB
    print("[OK] 导入成功")
    
    # 测试创建 FAISS 实例
    faiss_db = FAISSVectorDB()
    print("[OK] 创建 FAISSVectorDB 实例成功")
    
    # 测试基本操作
    chunks = ["这是第一个测试文档", "这是第二个测试文档", "这是第三个测试文档"]
    vectors = np.random.randn(3, 1536).astype(np.float32)
    
    faiss_db.add(chunks, vectors, ["test1.txt", "test2.txt", "test3.txt"])
    print("[OK] 添加文档块成功")
    
    print(f"文档块数量: {len(faiss_db)}")
    
    # 测试检索
    query_vector = np.random.randn(1, 1536).astype(np.float32)
    results = faiss_db.search(query_vector, top_k=2)
    print(f"[OK] 检索成功，返回 {len(results)} 条结果")
    
    # 测试混合检索
    results = faiss_db.hybrid_search("测试", query_vector, top_k=2)
    print(f"[OK] 混合检索成功，返回 {len(results)} 条结果")
    
    # 测试保存和加载
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        faiss_db.save(tmpdir)
        print("[OK] 保存成功")
        
        new_db = FAISSVectorDB()
        new_db.load(tmpdir)
        print(f"[OK] 加载成功，文档块数量: {len(new_db)}")
    
    print("\n✅ FAISS 向量数据库集成测试通过！")
    
except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试工厂模式
print("\n测试 VectorDBFactory...")
try:
    db = VectorDBFactory.create("faiss")
    print("[OK] VectorDBFactory.create('faiss') 成功")
    
    db2 = VectorDBFactory.create("memory")
    print("[OK] VectorDBFactory.create('memory') 成功")
    
    print("✅ VectorDBFactory 测试通过！")
except Exception as e:
    print(f"\n❌ VectorDBFactory 测试失败: {e}")
    sys.exit(1)
