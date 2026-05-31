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
    batch_size: int = 10,
) -> np.ndarray:
    """
    将文本列表批量转换为向量

    参数：
      texts:  要转换的文本列表
      model:  embedding 模型名
      api_key: DashScope API Key（默认从环境变量读取）
      batch_size: 每批处理的文本数量（API 限制最大 10）

    返回：
      numpy array，形状为 (len(texts), embedding_dim)
    """
    if not texts:
        raise ValueError("输入文本列表不能为空")

    # DashScope API 限制每批最多 10 条
    batch_size = min(batch_size, 10)
    
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    print(f"正在生成向量嵌入 (共 {len(texts)} 条，分 {total_batches} 批处理)...")
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        resp = TextEmbedding.call(
            model=model,
            input=batch_texts,
            api_key=api_key,
        )
        
        if resp.status_code != 200:
            raise RuntimeError(
                f"Embedding API 调用失败: [{resp.status_code}] {resp.message}"
            )
        
        embeddings = [
            item["embedding"]
            for item in resp.output["embeddings"]
        ]
        all_embeddings.extend(embeddings)
        
        if batch_num % 10 == 0 or batch_num == total_batches:
            print(f"  已处理 {batch_num}/{total_batches} 批")
    
    return np.array(all_embeddings, dtype=np.float32)


def embed_text(
    text: str,
    model: str = "text-embedding-v2",
    api_key: Optional[str] = None,
) -> np.ndarray:
    """单条文本嵌入的便捷函数 - 返回形状为 (1, dim) 的数组"""
    return embed_texts([text], model=model, api_key=api_key)
