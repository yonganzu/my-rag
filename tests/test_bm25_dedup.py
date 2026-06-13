"""
BM25 检索器去重功能测试

测试目标：
1. 验证 add_documents 可以正确添加新文档
2. 验证重复文档不会被添加（避免词频累积错误）
3. 验证混合文档（部分新 + 部分重复）可以正确处理
4. 验证搜索结果在去重后仍然正确
"""

import sys
import os

# 确保可以从项目根目录导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.vector_db.bm25_retriever import BM25Retriever


class TestBM25Dedup:
    """测试 BM25 检索器的去重功能"""

    def test_initial_empty(self):
        """初始化时应该为空"""
        retriever = BM25Retriever()
        assert retriever.is_empty() is True
        assert len(retriever) == 0

    def test_build_then_search(self):
        """基础构建和搜索测试"""
        retriever = BM25Retriever()
        docs = [
            "人工智能是计算机科学的一个分支",
            "机器学习是人工智能的重要组成部分",
            "深度学习是机器学习的一种方法",
        ]
        retriever.build(docs)

        assert len(retriever) == 3
        assert retriever.is_empty() is False

        results = retriever.search("人工智能", top_k=2)
        assert len(results) > 0
        # 第一个结果应该是包含"人工智能"的文档
        assert "人工智能" in results[0][2]

    def test_add_documents_new(self):
        """测试添加全新文档"""
        retriever = BM25Retriever()
        retriever.build(["文档一内容测试"])

        old_count = len(retriever)
        retriever.add_documents(["新文档内容", "另一个新文档"])

        assert len(retriever) == old_count + 2

    def test_add_documents_duplicate(self):
        """测试添加重复文档时应该被跳过"""
        retriever = BM25Retriever()
        docs = ["测试文档一内容", "测试文档二内容"]
        retriever.build(docs)
        old_count = len(retriever)

        # 添加完全重复的文档
        retriever.add_documents(["测试文档一内容"])
        assert len(retriever) == old_count  # 文档数不应增加

        # 搜索结果应该正常，词频不应异常
        results = retriever.search("测试", top_k=3)
        assert len(results) > 0

    def test_add_documents_mixed(self):
        """测试混合添加（部分新文档 + 部分重复）"""
        retriever = BM25Retriever()
        docs = ["原始文档一内容", "原始文档二内容"]
        retriever.build(docs, sources=["file1.txt", "file2.txt"])
        old_count = len(retriever)

        # 添加混合文档：一个新的 + 一个重复的
        retriever.add_documents(
            ["原始文档一内容", "全新文档三内容"],
            sources=["file1.txt", "file3.txt"]
        )

        # 只应该增加1个文档
        assert len(retriever) == old_count + 1

    def test_term_doc_freq_consistency(self):
        """验证去重后词频计算正确（不应该累积）"""
        retriever1 = BM25Retriever()
        retriever2 = BM25Retriever()

        docs = ["测试文档内容用于验证", "另一个测试内容"]
        retriever1.build(docs)

        # 第二种方式：先构建一个文档，再添加另一个
        retriever2.build([docs[0]])
        retriever2.add_documents([docs[1]])

        # 两者的 term_doc_freq 应该一致
        assert retriever1._term_doc_freq == retriever2._term_doc_freq

        # 搜索结果评分也应该近似
        r1 = retriever1.search("测试", top_k=2)
        r2 = retriever2.search("测试", top_k=2)
        scores1 = [r[1] for r in r1]
        scores2 = [r[1] for r in r2]
        for s1, s2 in zip(scores1, scores2):
            assert abs(s1 - s2) < 0.001, f"分数差异过大: {s1} vs {s2}"

    def test_add_with_sources(self):
        """测试带来源信息的添加"""
        retriever = BM25Retriever()
        retriever.build(["文档一", "文档二"], ["src1.txt", "src2.txt"])

        retriever.add_documents(["文档三"], ["src3.txt"])

        assert len(retriever.sources) == 3
        assert retriever.sources[-1] == "src3.txt"

    def test_add_without_sources(self):
        """测试不带来源信息的添加，应该自动填充默认值"""
        retriever = BM25Retriever()
        retriever.build(["文档一"])

        retriever.add_documents(["新文档"])  # 不传 sources

        assert len(retriever.sources) == 2
        # 新添加的应该有默认来源
        assert retriever.sources[-1] == "未知来源"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
