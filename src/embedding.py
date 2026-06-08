"""
文本嵌入（Embedding）模块

什么是 Embedding？
  - 将文本转换成固定维度的向量（数字数组）
  - 语义相近的文本，向量在高维空间中距离也近
  - RAG 的核心：将文档和问题都转成向量，用向量相似度检索

支持的后端：
  - DashScope API：通义千问的 text-embedding-v2，中文优化
  - 本地模型：Sentence-BERT 系列模型

工程要点：
  - API 调用需要网络 IO，要做好错误处理
  - embedding 有速率限制（RPM），生产环境需加退避重试
  - 返回的向量维度取决于模型（text-embedding-v2 是 1536 维）
"""

from typing import List, Optional

import numpy as np

from src.config import config


# ── 本地模型缓存 ────────────────────────────────────────────────
_local_embedding_model = None


def _get_local_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """获取本地嵌入模型（懒加载）"""
    global _local_embedding_model
    if _local_embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            print(f"[Embedding] 加载本地模型: {model_name}")
            _local_embedding_model = SentenceTransformer(model_name)
            print(f"[Embedding] 本地模型加载完成，向量维度: {_local_embedding_model.get_sentence_embedding_dimension()}")
        except ImportError:
            raise ImportError("请安装 sentence-transformers 以使用本地嵌入模型: pip install sentence-transformers")
    return _local_embedding_model


def _embed_texts_local(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> np.ndarray:
    """
    使用本地模型进行文本嵌入
    
    参数：
      texts: 要转换的文本列表
      model_name: Sentence-BERT 模型名称
      batch_size: 每批处理的文本数量
    
    返回：
      numpy array，形状为 (len(texts), embedding_dim)
    """
    model = _get_local_embedding_model(model_name)
    
    print(f"[Embedding] 使用本地模型 {model_name}，正在嵌入 {len(texts)} 条文本...")
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    
    return embeddings.astype(np.float32)


def _embed_texts_dashscope(
    texts: List[str],
    model: str = "text-embedding-v2",
    api_key: Optional[str] = None,
    batch_size: int = 10,
) -> np.ndarray:
    """
    使用 DashScope API 进行文本嵌入
    
    参数：
      texts: 要转换的文本列表
      model: embedding 模型名
      api_key: DashScope API Key（默认从环境变量读取）
      batch_size: 每批处理的文本数量（API 限制最大 10）
    
    返回：
      numpy array，形状为 (len(texts), embedding_dim)
    """
    from dashscope import TextEmbedding
    
    if not texts:
        raise ValueError("输入文本列表不能为空")

    batch_size = min(batch_size, 10)
    
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    print(f"[Embedding] 使用 DashScope API ({model})，正在生成向量嵌入 (共 {len(texts)} 条，分 {total_batches} 批处理)...")
    
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


def embed_texts(
    texts: List[str],
    model: str = "text-embedding-v2",
    api_key: Optional[str] = None,
    batch_size: int = 10,
    embedding_type: Optional[str] = None,
) -> np.ndarray:
    """
    将文本列表批量转换为向量
    
    参数：
      texts: 要转换的文本列表
      model: embedding 模型名
      api_key: DashScope API Key（默认从环境变量读取）
      batch_size: 每批处理的文本数量
      embedding_type: 嵌入类型 ("dashscope" 或 "local")，默认为配置值
    
    返回：
      numpy array，形状为 (len(texts), embedding_dim)
    """
    if embedding_type is None:
        embedding_type = config.embedding_type
    
    if embedding_type == "local":
        local_model = model if model != "text-embedding-v2" else config.embedding_local_model
        return _embed_texts_local(texts, model_name=local_model, batch_size=batch_size)
    else:
        return _embed_texts_dashscope(texts, model=model, api_key=api_key, batch_size=batch_size)


def embed_text(
    text: str,
    model: str = "text-embedding-v2",
    api_key: Optional[str] = None,
    embedding_type: Optional[str] = None,
) -> np.ndarray:
    """单条文本嵌入的便捷函数 - 返回形状为 (1, dim) 的数组"""
    return embed_texts(
        [text], 
        model=model, 
        api_key=api_key,
        embedding_type=embedding_type
    )
