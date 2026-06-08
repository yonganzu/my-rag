"""
测试 Qwen3-Reranker API
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

local_packages = project_root / "local_packages" / "Lib" / "site-packages"
if local_packages.exists():
    sys.path.insert(0, str(local_packages))

from src.config import config

print("=" * 60)
print("测试 Qwen3-Reranker API")
print("=" * 60)

if not config.dashscope_api_key:
    print("错误：请先设置 DASHSCOPE_API_KEY 环境变量")
    sys.exit(1)

print("当前配置:")
print("  rerank_method: %s" % config.rerank_method)
print("  reranker_type: %s" % config.reranker_type)
print("  qwen3_reranker_model: %s" % config.qwen3_reranker_model)
print()

try:
    from dashscope import TextReRank
    
    print("测试 DashScope TextReRank API...")
    
    query = "什么是人工智能？"
    documents = [
        "人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，致力于研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统。",
        "机器学习是人工智能的核心技术之一，它使计算机能够从数据中学习并改进其性能，而无需进行明确编程。",
        "深度学习是机器学习的一个子领域，使用多层神经网络来模拟人脑的学习过程。",
        "自然语言处理（NLP）是人工智能的一个重要应用领域，使计算机能够理解、处理和生成人类语言。",
        "计算机视觉是人工智能的另一个重要领域，使计算机能够理解和分析图像和视频内容。",
    ]
    
    print("查询: %s" % query)
    print("候选文档数量: %d" % len(documents))
    print()
    
    resp = TextReRank.call(
        model=config.qwen3_reranker_model,
        query=query,
        documents=documents,
        api_key=config.dashscope_api_key,
    )
    
    if resp.status_code != 200:
        print("API 调用失败: [%d] %s" % (resp.status_code, resp.message))
        sys.exit(1)
    
    print("API 调用成功！")
    print()
    print("重排序结果:")
    print("-" * 60)
    
    results = []
    for item in resp.output["results"]:
        idx = int(item["index"])
        score = float(item["relevance_score"])
        results.append((idx, score))
    
    results.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (idx, score) in enumerate(results):
        print("排名 %d:" % (rank+1))
        print("  分数: %.4f" % score)
        print("  内容: %s..." % documents[idx][:100])
        print()
    
    print("=" * 60)
    print("测试完成！")
    
except ImportError as e:
    print("导入失败: %s" % e)
    print("请安装 dashscope: pip install dashscope")
    sys.exit(1)
except Exception as e:
    print("测试失败: %s" % e)
    import traceback
    traceback.print_exc()
    sys.exit(1)