"""
RAG 评估脚本 - 输出到文件，包含详细分类分析
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import config
from src.rag_pipeline import RAGPipeline
from tests.test_dataset import test_dataset


def main():
    output_lines = []
    output_lines.append("=== RAG 系统评估 ===")
    
    # 初始化 RAG 管道
    rag = RAGPipeline()
    
    # 加载知识库
    try:
        rag.load_knowledge_base()
        output_lines.append(f"[INFO] 知识库加载成功，共 {len(rag.vector_store)} 个文档块")
    except Exception as e:
        output_lines.append(f"[ERROR] 加载知识库失败: {e}")
        write_results(output_lines)
        return
    
    output_lines.append(f"测试样本数: {len(test_dataset)}")
    
    # 记录统计
    category_stats = {}  # 按类别统计
    total_recall = 0
    total_hits = 0
    total_questions = 0
    total_errors = 0
    
    for idx, sample in enumerate(test_dataset):
        question = sample["question"]
        expected_sources = sample["expected_sources"]
        category = sample["category"]
        
        output_lines.append(f"\n[{idx+1}/{len(test_dataset)}] 问题: {question}")
        output_lines.append(f"  类别: {category}")
        
        try:
            # 获取检索结果
            _, sources = rag.retriever.retrieve(
                question,
                use_rerank=False,
                use_query_rewrite=False,
            )
            
            output_lines.append(f"  检索到的来源: {sources}")
            output_lines.append(f"  期望的来源: {expected_sources}")
            
            # 检查是否命中
            retrieved_set = set(sources) if sources else set()
            expected_set = set(expected_sources)
            hit = bool(retrieved_set & expected_set)
            
            if hit:
                output_lines.append(f"  [HIT] 检索命中")
                total_hits += 1
            else:
                output_lines.append(f"  [MISS] 未命中")
            
            # 计算召回率
            if expected_set:
                recall = len(retrieved_set & expected_set) / len(expected_set)
            else:
                recall = 0.0
            total_recall += recall
            total_questions += 1
            
            # 按类别统计
            if category not in category_stats:
                category_stats[category] = {"count": 0, "hits": 0, "recall_sum": 0}
            category_stats[category]["count"] += 1
            category_stats[category]["recall_sum"] += recall
            if hit:
                category_stats[category]["hits"] += 1
            
        except Exception as e:
            output_lines.append(f"  [ERROR] 错误: {e}")
            import traceback
            output_lines.append(f"  {traceback.format_exc()}")
            total_errors += 1
    
    # 输出总体统计结果
    output_lines.append("\n" + "=" * 60)
    output_lines.append("评估结果 - 总体")
    output_lines.append("=" * 60)
    output_lines.append(f"总问题数: {len(test_dataset)}")
    output_lines.append(f"有效问题数: {total_questions}")
    output_lines.append(f"错误数: {total_errors}")
    if total_questions > 0:
        output_lines.append(f"命中数: {total_hits}")
        output_lines.append(f"检索准确率: {total_hits/total_questions*100:.2f}%")
        output_lines.append(f"检索召回率: {total_recall/total_questions*100:.2f}%")
    else:
        output_lines.append("没有成功执行任何测试")
    
    # 输出分类统计结果
    if category_stats:
        output_lines.append("\n" + "=" * 60)
        output_lines.append("评估结果 - 按类别")
        output_lines.append("=" * 60)
        for category, stats in sorted(category_stats.items()):
            count = stats["count"]
            hits = stats["hits"]
            recall_sum = stats["recall_sum"]
            if count > 0:
                acc = hits / count * 100
                recall = recall_sum / count * 100
                output_lines.append(f"类别: {category}")
                output_lines.append(f"  问题数: {count}")
                output_lines.append(f"  准确率: {acc:.2f}%")
                output_lines.append(f"  召回率: {recall:.2f}%")
    
    # 写入文件
    write_results(output_lines)
    print(f"评估完成！查看 {project_root / 'tests' / 'evaluation_results.txt'}")


def write_results(lines):
    output_path = project_root / "tests" / "evaluation_results.txt"
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"结果已保存到: {output_path}")
    except Exception as e:
        print(f"写入文件失败: {e}")
        print("=== 尝试打印结果 ===")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
