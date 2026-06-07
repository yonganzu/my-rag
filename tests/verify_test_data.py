
"""
测试脚本，验证测试数据是否正确加载
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.test_dataset import test_dataset

print(f"测试数据集总数: {len(test_dataset)}")

# 查看前5个和最后5个问题
print("\n前5个问题:")
for i, item in enumerate(test_dataset[:5], 1):
    print(f"{i}. {item['question']}")

print("\n后5个问题:")
for i, item in enumerate(test_dataset[-5:], 1):
    print(f"{len(test_dataset)-4 + i}. {item['question']}")

# 按类别统计
print("\n按类别统计:")
category_counts = {}
for item in test_dataset:
    cat = item['category']
    category_counts[cat] = category_counts.get(cat, 0) + 1

for cat, count in category_counts.items():
    print(f"  {cat}: {count}")
