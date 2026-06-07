"""
RAG 系统评估脚本

评估指标：
1. 检索召回率 (Retrieval Recall)：检索到的文档中包含期望来源的比例
2. MRR@3 (Mean Reciprocal Rank)：期望来源在检索结果中的排名倒数均值
3. 回答准确率 (Answer Accuracy)：基于语义相似度的回答质量评估
4. 引用准确率 (Citation Accuracy)：回答中正确标注来源的比例

使用方法：
python tests/evaluate_rag.py
"""

import json
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.embedding import embed_text
from src.rag_pipeline import RAGPipeline
from tests.test_dataset import test_dataset


def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """计算两个向量的余弦相似度"""
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def evaluate_rag_pipeline(rag: RAGPipeline, dataset: list) -> dict:
    """评估 RAG 系统性能"""
    results = []
    total_recall = 0.0
    total_mrr = 0.0
    total_similarity = 0.0
    total_citation_acc = 0.0
    
    print(f"正在评估 {len(dataset)} 个测试样本...\n")
    
    for i, sample in enumerate(dataset, 1):
        question = sample["question"]
        expected_sources = sample["expected_sources"]
        expected_answer = sample["expected_answer"]
        
        print(f"[{i}/{len(dataset)}] 问题: {question}")
        
        try:
            # 获取 RAG 回答和检索到的上下文
            answer, contexts = rag.answer(
                question,
                use_rerank=config.use_rerank,
                use_query_rewrite=config.use_query_rewrite,
                show_citations=config.show_citations,
            )
            
            # 获取检索到的来源（从上下文推断或直接获取）
            # 由于当前实现没有直接返回来源，我们通过检索器获取
            _, sources = rag.retriever.retrieve(
                question,
                use_rerank=config.use_rerank,
                use_query_rewrite=config.use_query_rewrite,
            )
            
            # 1. 计算检索召回率
            retrieved_sources = set(sources) if sources else set()
            expected_source_set = set(expected_sources)
            hit_count = len(retrieved_sources & expected_source_set)
            recall = hit_count / len(expected_source_set) if expected_source_set else 0.0
            total_recall += recall
            
            # 2. 计算 MRR@3
            mrr = 0.0
            for rank, source in enumerate(sources[:3], 1):
                if source in expected_source_set:
                    mrr = 1.0 / rank
                    break
            total_mrr += mrr
            
            # 3. 计算回答语义相似度
            answer_embedding = embed_text(answer)
            expected_embedding = embed_text(expected_answer)
            similarity = calculate_cosine_similarity(answer_embedding, expected_embedding)
            total_similarity += similarity
            
            # 4. 计算引用准确率
            citation_acc = 0.0
            if expected_sources:
                for expected_source in expected_sources:
                    if expected_source in answer:
                        citation_acc += 1.0
                citation_acc /= len(expected_sources)
            total_citation_acc += citation_acc
            
            # 记录结果
            results.append({
                "question": question,
                "answer": answer,
                "expected_answer": expected_answer,
                "retrieved_sources": sources,
                "expected_sources": expected_sources,
                "recall": recall,
                "mrr": mrr,
                "similarity": similarity,
                "citation_acc": citation_acc,
                "category": sample["category"]
            })
            
            print(f"   召回率: {recall:.2f} | MRR: {mrr:.2f} | 相似度: {similarity:.2f} | 引用准确率: {citation_acc:.2f}")
            
        except Exception as e:
            print(f"   ❌ 错误: {e}")
            results.append({
                "question": question,
                "error": str(e),
                "recall": 0.0,
                "mrr": 0.0,
                "similarity": 0.0,
                "citation_acc": 0.0,
                "category": sample["category"]
            })
    
    # 计算总体指标
    n_valid = len([r for r in results if "error" not in r])
    if n_valid == 0:
        return {"error": "所有测试样本都失败"}
    
    overall_metrics = {
        "total_samples": len(dataset),
        "valid_samples": n_valid,
        "retrieval_recall": total_recall / n_valid,
        "mrr_at_3": total_mrr / n_valid,
        "answer_similarity": total_similarity / n_valid,
        "citation_accuracy": total_citation_acc / n_valid,
        "results": results
    }
    
    return overall_metrics


def print_evaluation_report(metrics: dict):
    """打印评估报告"""
    print("\n" + "=" * 70)
    print("                    RAG 系统评估报告")
    print("=" * 70)
    print(f"测试样本总数: {metrics['total_samples']}")
    print(f"有效样本数: {metrics['valid_samples']}")
    print("-" * 70)
    print(f"检索召回率 (Retrieval Recall): {metrics['retrieval_recall']:.4f}")
    print(f"MRR@3 (Mean Reciprocal Rank): {metrics['mrr_at_3']:.4f}")
    print(f"回答语义相似度: {metrics['answer_similarity']:.4f}")
    print(f"引用准确率 (Citation Accuracy): {metrics['citation_accuracy']:.4f}")
    print("=" * 70)
    
    # 按类别分组统计
    categories = {}
    for result in metrics["results"]:
        if "error" in result:
            continue
        cat = result["category"]
        if cat not in categories:
            categories[cat] = {"count": 0, "recall": 0, "mrr": 0, "similarity": 0}
        categories[cat]["count"] += 1
        categories[cat]["recall"] += result["recall"]
        categories[cat]["mrr"] += result["mrr"]
        categories[cat]["similarity"] += result["similarity"]
    
    print("\n按类别分组统计:")
    print("-" * 70)
    for cat, stats in categories.items():
        print(f"{cat}:")
        print(f"  样本数: {stats['count']}")
        print(f"  召回率: {stats['recall']/stats['count']:.4f}")
        print(f"  MRR@3: {stats['mrr']/stats['count']:.4f}")
        print(f"  相似度: {stats['similarity']/stats['count']:.4f}")
    
    # 保存详细结果
    output_path = Path(__file__).parent / "evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {output_path}")


def main():
    """主函数"""
    # 初始化 RAG 管道
    print("初始化 RAG 管道...")
    rag = RAGPipeline()
    vector_db_path = rag.db_path
    
    # 加载或构建知识库
    if vector_db_path.exists() and any(vector_db_path.iterdir()):
        print(f"加载已有知识库: {vector_db_path}")
        rag.load_knowledge_base()
    else:
        print("错误：未找到向量数据库，请先运行 main.py 构建知识库")
        return
    
    # 执行评估
    metrics = evaluate_rag_pipeline(rag, test_dataset)
    
    if "error" in metrics:
        print(f"评估失败: {metrics['error']}")
        return
    
    # 打印报告
    print_evaluation_report(metrics)


if __name__ == "__main__":
    main()
