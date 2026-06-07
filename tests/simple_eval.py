"""
简单的 RAG 评估脚本
"""

import sys
import traceback
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import config
from src.rag_pipeline import RAGPipeline
from tests.test_dataset import test_dataset


def main():
    print("=== RAG 系统评估 ===")
    
    # 初始化 RAG 管道
    rag = RAGPipeline()
    
    # 加载知识库
    try:
        rag.load_knowledge_base()
        print("[INFO] 知识库加载成功，共 %d 个文档块" % len(rag.vector_store))
    except Exception as e:
        print("[ERROR] 加载知识库失败: %s" % e)
        traceback.print_exc()
        return
    
    # 测试样本
    print("测试样本数: %d" % len(test_dataset))
    
    # 记录统计
    total_recall = 0
    total_hits = 0
    total_questions = 0
    total_errors = 0
    
    for idx, sample in enumerate(test_dataset):
        question = sample["question"]
        expected_sources = sample["expected_sources"]
        
        print("\n[%d/%d] 问题: %s" % (idx+1, len(test_dataset), question))
        
        try:
            # 获取检索结果
            _, sources = rag.retriever.retrieve(
                question,
                use_rerank=False,
                use_query_rewrite=False,
            )
            
            print("检索到的来源: %s" % str(sources))
            print("期望的来源: %s" % str(expected_sources))
            
            # 检查是否命中
            retrieved_set = set(sources) if sources else set()
            expected_set = set(expected_sources)
            hit = bool(retrieved_set & expected_set)
            
            if hit:
                print("[HIT] 检索命中")
                total_hits += 1
            else:
                print("[MISS] 未命中")
            
            # 计算召回率
            if expected_set:
                recall = len(retrieved_set & expected_set) / len(expected_set)
            else:
                recall = 0.0
            total_recall += recall
            total_questions += 1
            
        except Exception as e:
            print("[ERROR] 错误: %s" % e)
            traceback.print_exc()
            total_errors += 1
    
    # 输出统计结果
    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)
    print("总问题数: %d" % len(test_dataset))
    print("有效问题数: %d" % total_questions)
    print("错误数: %d" % total_errors)
    if total_questions > 0:
        print("命中数: %d" % total_hits)
        print("检索准确率: %.1f%%" % (total_hits/total_questions*100))
        print("检索召回率: %.1f%%" % (total_recall/total_questions*100))
    else:
        print("没有成功执行任何测试")


if __name__ == "__main__":
    main()
