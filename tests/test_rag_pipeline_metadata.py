"""
RAGPipeline 元数据管理测试

测试目标：
1. 验证 _doc_metadata 初始化为空字典而非 None
2. 验证 build_knowledge_base 正确设置 _doc_metadata
3. 验证 add_documents 正确合并 _doc_metadata（即使先 add 再 build）
4. 验证 check_documents_update 在 metadata 为空时能正确识别所有文件为新
5. 验证 answer 方法接收 user_role 参数并传递给 retriever
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Mock 掉 embedding 相关的网络调用，避免测试时需要 API Key


def mock_embed_texts(texts, model=None, **kwargs):
    """Mock 的 embedding 函数：返回简单的随机向量"""
    import numpy as np
    np.random.seed(42)
    dim = 64  # 小维度加速测试
    return np.random.randn(len(texts), dim).astype(np.float32)


def mock_embed_text(text, model=None, **kwargs):
    """Mock 的单条 embedding 函数"""
    import numpy as np
    np.random.seed(42)
    dim = 64
    return np.random.randn(1, dim).astype(np.float32)


def mock_llm_call(model, messages, **kwargs):
    """Mock 的 LLM 调用"""
    return "这是一个测试回答"


class TestRAGPipelineMetadata:
    """测试 RAGPipeline 元数据管理"""

    @patch('src.rag_pipeline.embed_texts', side_effect=mock_embed_texts)
    @patch('src.rag_pipeline.llm_call', side_effect=mock_llm_call)
    @patch('src.retrieval.embed_texts', side_effect=mock_embed_texts)
    @patch('src.retrieval.embed_text', side_effect=mock_embed_text)
    def test_initial_metadata_not_none(self, *mocks):
        """测试初始化时 _doc_metadata 应该是字典而非 None"""
        from src.rag_pipeline import RAGPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "vector_db")
            rag = RAGPipeline(db_path=db_path, db_type="memory")

            assert rag._doc_metadata is not None
            assert isinstance(rag._doc_metadata, dict)
            assert len(rag._doc_metadata) == 0

    @patch('src.rag_pipeline.embed_texts', side_effect=mock_embed_texts)
    @patch('src.rag_pipeline.llm_call', side_effect=mock_llm_call)
    @patch('src.retrieval.embed_texts', side_effect=mock_embed_texts)
    @patch('src.retrieval.embed_text', side_effect=mock_embed_text)
    def test_build_sets_metadata(self, *mocks):
        """测试 build_knowledge_base 后 _doc_metadata 应该是字典"""
        from src.rag_pipeline import RAGPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "vector_db")
            rag = RAGPipeline(db_path=db_path, db_type="memory")

            chunks = ["测试文档一的内容", "测试文档二的内容"]
            metadata = {
                "doc1.txt": {"mtime": 1000.0, "size": 100},
                "doc2.txt": {"mtime": 2000.0, "size": 200},
            }
            sources = ["doc1.txt|internal", "doc2.txt|internal"]

            rag.build_knowledge_base(chunks, doc_metadata=metadata, chunk_sources=sources)

            assert rag._doc_metadata is not None
            assert isinstance(rag._doc_metadata, dict)
            assert "doc1.txt" in rag._doc_metadata
            assert "doc2.txt" in rag._doc_metadata

    @patch('src.rag_pipeline.embed_texts', side_effect=mock_embed_texts)
    @patch('src.rag_pipeline.llm_call', side_effect=mock_llm_call)
    @patch('src.retrieval.embed_texts', side_effect=mock_embed_texts)
    @patch('src.retrieval.embed_text', side_effect=mock_embed_text)
    def test_add_documents_without_prior_build(self, *mocks):
        """测试 add_documents 能在没有先 build 的情况下工作（_doc_metadata 初始为空 dict）"""
        from src.rag_pipeline import RAGPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "vector_db")
            rag = RAGPipeline(db_path=db_path, db_type="memory")

            # 直接 add，不先 build
            chunks = ["增量测试文档内容"]
            new_metadata = {
                "new_doc.txt": {"mtime": 3000.0, "size": 150},
            }
            sources = ["new_doc.txt|internal"]

            rag.add_documents(chunks, doc_metadata=new_metadata, chunk_sources=sources)

            # 验证元数据已正确合并
            assert "new_doc.txt" in rag._doc_metadata
            assert rag._doc_metadata["new_doc.txt"]["size"] == 150

    @patch('src.rag_pipeline.embed_texts', side_effect=mock_embed_texts)
    @patch('src.rag_pipeline.llm_call', side_effect=mock_llm_call)
    @patch('src.retrieval.embed_texts', side_effect=mock_embed_texts)
    @patch('src.retrieval.embed_text', side_effect=mock_embed_text)
    def test_add_documents_merges_metadata(self, *mocks):
        """测试 add_documents 能正确合并元数据"""
        from src.rag_pipeline import RAGPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "vector_db")
            rag = RAGPipeline(db_path=db_path, db_type="memory")

            # 第一步：构建知识库
            chunks1 = ["文档一内容"]
            metadata1 = {"doc1.txt": {"mtime": 1000.0, "size": 100}}
            rag.build_knowledge_base(chunks1, doc_metadata=metadata1, chunk_sources=["doc1.txt|internal"])

            # 第二步：增量添加
            chunks2 = ["文档二内容"]
            metadata2 = {"doc2.txt": {"mtime": 2000.0, "size": 200}}
            rag.add_documents(chunks2, doc_metadata=metadata2, chunk_sources=["doc2.txt|internal"])

            # 验证：两个文档的元数据都应该存在
            assert "doc1.txt" in rag._doc_metadata
            assert "doc2.txt" in rag._doc_metadata
            assert len(rag._doc_metadata) == 2

    def test_check_documents_update_empty_metadata(self):
        """测试 check_documents_update 在 metadata 为空时应该将所有文件视为新增"""
        from src.rag_pipeline import RAGPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建测试文档
            docs_folder = os.path.join(tmpdir, "documents")
            os.makedirs(docs_folder, exist_ok=True)
            test_file = os.path.join(docs_folder, "test.txt")
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("测试文档内容")

            rag = RAGPipeline(db_path=os.path.join(tmpdir, "vector_db"), db_type="memory")
            # _doc_metadata 初始是空字典

            has_update, new_files, modified_files = rag.check_documents_update(docs_folder)

            assert has_update is True
            assert "test.txt" in new_files
            assert len(modified_files) == 0

    def test_check_documents_update_with_existing_metadata(self):
        """测试有旧元数据时能正确识别修改/新增/删除的文件"""
        from src.rag_pipeline import RAGPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            docs_folder = os.path.join(tmpdir, "documents")
            os.makedirs(docs_folder, exist_ok=True)

            # 创建文档
            test_file1 = os.path.join(docs_folder, "doc1.txt")
            test_file2 = os.path.join(docs_folder, "doc2.txt")
            with open(test_file1, "w", encoding="utf-8") as f:
                f.write("文档一内容")
            with open(test_file2, "w", encoding="utf-8") as f:
                f.write("文档二内容")

            rag = RAGPipeline(db_path=os.path.join(tmpdir, "vector_db"), db_type="memory")

            # 设置旧元数据（doc1 存在但修改时间不同，doc3 不存在于文件夹）
            stat1 = os.stat(test_file1)
            rag._doc_metadata = {
                "doc1.txt": {"mtime": stat1.st_mtime - 1000, "size": stat1.st_size},  # 旧 mtime
                "doc3.txt": {"mtime": 100, "size": 50},  # 已删除
            }

            has_update, new_files, modified_files = rag.check_documents_update(docs_folder)

            # doc2.txt 应为新增，doc1.txt 应为修改
            assert has_update is True
            assert "doc2.txt" in new_files
            assert "doc1.txt" in modified_files

    @patch('src.rag_pipeline.embed_texts', side_effect=mock_embed_texts)
    @patch('src.rag_pipeline.llm_call', side_effect=mock_llm_call)
    @patch('src.retrieval.embed_texts', side_effect=mock_embed_texts)
    @patch('src.retrieval.embed_text', side_effect=mock_embed_text)
    def test_answer_accepts_user_role(self, *mocks):
        """测试 answer 方法接收 user_role 参数"""
        from src.rag_pipeline import RAGPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "vector_db")
            rag = RAGPipeline(db_path=db_path, db_type="memory")

            chunks = ["测试文档内容"]
            sources = ["test.txt|internal"]
            rag.build_knowledge_base(chunks, chunk_sources=sources)

            # 验证 answer 接受 user_role 参数
            answer, contexts = rag.answer(
                "测试问题",
                use_rerank=False,
                use_query_rewrite=False,
                user_role="admin"
            )
            assert isinstance(answer, str)

            # 验证普通用户角色也能工作
            answer2, contexts2 = rag.answer(
                "测试问题",
                use_rerank=False,
                use_query_rewrite=False,
                user_role="user"
            )
            assert isinstance(answer2, str)

    @patch('src.rag_pipeline.embed_texts', side_effect=mock_embed_texts)
    @patch('src.rag_pipeline.llm_call', side_effect=mock_llm_call)
    @patch('src.retrieval.embed_texts', side_effect=mock_embed_texts)
    @patch('src.retrieval.embed_text', side_effect=mock_embed_text)
    def test_build_with_none_metadata(self, *mocks):
        """测试 build_knowledge_base 在 doc_metadata=None 时不会出问题"""
        from src.rag_pipeline import RAGPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "vector_db")
            rag = RAGPipeline(db_path=db_path, db_type="memory")

            # 不传 doc_metadata
            chunks = ["测试文档"]
            rag.build_knowledge_base(chunks, chunk_sources=["test.txt|internal"])

            # _doc_metadata 应该是空字典而不是 None
            assert rag._doc_metadata is not None
            assert isinstance(rag._doc_metadata, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
