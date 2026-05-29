"""
文本嵌入（Embedding）模块

什么是 Embedding？
  - 将文本转换成固定维度的向量（数字数组）
  - 语义相近的文本，向量在高维空间中距离也近
  - RAG 的核心：将文档和问题都转成向量，用向量相似度检索

为什么用 DashScope 的 embedding API？
  - 通义千问的 text-embedding-v2 是中文优化模型
  - 无需自己部署 embedding 模型，API 调用即可
  - 企业级稳定性，适合生产环境

工程要点：
  - API 调用需要网络 IO，要做好错误处理
  - embedding 有速率限制（RPM），生产环境需加退避重试
  - 返回的向量维度取决于模型（text-embedding-v2 是 1536 维）
"""

from typing import List, Optional

import numpy as np
from dashscope import TextEmbedding


def embed_texts(
    texts: List[str],
    model: str = "text-embedding-v2",
    api_key: Optional[str] = None,
) -> np.ndarray:
    """
    将文本列表批量转换为向量

    参数：
      texts:  要转换的文本列表
      model:  embedding 模型名
      api_key: DashScope API Key（默认从环境变量读取）

    返回：
      numpy array，形状为 (len(texts), embedding_dim)
    """
    if not texts:
        raise ValueError("输入文本列表不能为空")

    # ── 调用 DashScope Embedding API ───────────────────────────
    # TextEmbedding.call 是 dashscope SDK 提供的同步接口
    # batch_size 控制在一次请求中发送多少文本
    resp = TextEmbedding.call(
        model=model,
        input=texts,
        api_key=api_key,
        batch_size=25,  # 每批最多 25 条，超过会自动分批
    )

    # ── 错误处理 ──────────────────────────────────────────────
    # API 返回状态码，非 200 表示失败
    if resp.status_code != 200:
        raise RuntimeError(
            f"Embedding API 调用失败: [{resp.status_code}] {resp.message}"
        )

    # ── 提取向量 ──────────────────────────────────────────────
    # resp.output["embeddings"] 是列表，每个元素包含 embedding 值
    # 按原始顺序组装成 numpy 矩阵
    embeddings = [
        item["embedding"]
        for item in resp.output["embeddings"]
    ]

    # np.array() 将 Python 列表转为 numpy 数组
    # 这是后续向量计算的基础数据结构
    return np.array(embeddings, dtype=np.float32)


def embed_text(
    text: str,
    model: str = "text-embedding-v2",
    api_key: Optional[str] = None,
) -> np.ndarray:
    """单条文本嵌入的便捷函数 - 返回形状为 (1, dim) 的数组"""
    return embed_texts([text], model=model, api_key=api_key)
