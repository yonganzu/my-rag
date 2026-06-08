"""
测试 BGE-Reranker-v2-m3
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
print("测试 BGE-Reranker-v2-m3")
print("=" * 60)

print("当前配置:")
print("  rerank_method: %s" % config.rerank_method)
print("  reranker_type: %s" % config.reranker_type)
print("  bge_reranker_model: %s" % config.bge_reranker_model)
print()

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    
    print("正在加载 BGE-Reranker 模型...")
    model_name = config.bge_reranker_model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    print("模型加载成功！")
    
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
    
    # 准备输入对
    pairs = [[query, ctx] for ctx in documents]
    
    # 推理
    scores = []
    batch_size = 8
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i+batch_size]
        
        encoding = tokenizer(
            batch_pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
            if logits.shape[1] >= 2:
                # 对于二分类模型，取正类的概率
                batch_scores = torch.sigmoid(logits[:, 1]).tolist()
            else:
                # 对于回归模型或单输出模型
                batch_scores = logits.squeeze().tolist()
            scores.extend(batch_scores)
    
    # 按分数排序
    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 输出重排序详情
    print("重排序结果:")
    print("-" * 60)
    for rank, (idx, score) in enumerate(indexed_scores):
        print("排名 %d:" % (rank+1))
        print("  分数: %.4f" % score)
        print("  内容: %s..." % documents[idx][:100])
        print()
    
    print("=" * 60)
    print("测试完成！")
    
except ImportError as e:
    print("导入失败: %s" % e)
    print("请安装 transformers 和 torch: pip install transformers torch")
    sys.exit(1)
except Exception as e:
    print("测试失败: %s" % e)
    import traceback
    traceback.print_exc()
    sys.exit(1)
