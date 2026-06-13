"""
配置管理模块

为什么要独立一个 config.py？
  - 所有配置集中管理，修改配置只需要改一处（或 .env 文件）
  - 代码中不硬编码 API Key、模型名等敏感/易变信息
  - 团队协作时通过 .env.example 共享配置模板，而不泄露密钥
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv


# ── 加载 .env 文件 ─────────────────────────────────────────────
# python-dotenv 将 .env 中的变量注入 os.environ
# 优先级：已存在的系统环境变量 > .env 文件
# 这样用户可以在 shell 配置中 export DASHSCOPE_API_KEY，.env 只放非敏感配置
load_dotenv()


@dataclass
class Config:
    """类型化的配置对象，比直接读 os.environ 更安全、可发现"""

    # ── DashScope API Key（从 .env 文件读取）────────────────────
    # 请在 .env 文件中设置 DASHSCOPE_API_KEY=sk-xxx
    dashscope_api_key: str = os.environ.get("DASHSCOPE_API_KEY", "")

    # ── LLM 配置（通过 LLM_TYPE 切换 API/本地模式）──────────────────
    # LLM_TYPE 可选值:
    #   - dashscope: 使用 DashScope API（需要 DASHSCOPE_API_KEY）
    #   - ollama: 使用本地 Ollama（需要先安装并运行 ollama）
    #   - local: 使用本地 Transformers 模型
    llm_type: str = os.environ.get("LLM_TYPE", "dashscope")
    llm_model: str = os.environ.get("LLM_MODEL", "qwen-plus")
    llm_base_url: str = os.environ.get(
        "LLM_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # ── Embedding 配置（通过 EMBEDDING_TYPE 切换 API/本地模式）────────
    # EMBEDDING_TYPE 可选值:
    #   - dashscope: 使用 DashScope API（需要 DASHSCOPE_API_KEY）
    #   - local: 使用本地 Sentence-BERT 模型
    embedding_type: str = os.environ.get("EMBEDDING_TYPE", "dashscope")
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "text-embedding-v2")
    embedding_local_model: str = os.environ.get("EMBEDDING_LOCAL_MODEL", "all-MiniLM-L6-v2")

    # ── 文本分块配置 ────────────────────────────────────────────
    # chunk_size: 每块字符数（不是 token 数，但足够了演示）
    # chunk_overlap: 块与块之间的重叠字符数，避免切碎关键上下文
    chunk_size: int = 500
    chunk_overlap: int = 100

    # ── 检索配置 ────────────────────────────────────────────────
    top_k: int = 3  # 检索时返回最相似的 top-k 个块
    use_rerank: bool = True  # 是否使用重排序
    rerank_factor: int = 3    # rerank 时先获取的候选数量倍数
    
    # 重排序方法（通过 RERANK_METHOD 切换）:
    #   - none: 不使用重排序
    #   - vector: 向量相似度重排序（免费，快速）
    #   - keyword: 关键词匹配重排序（免费，快速）
    #   - llm: LLM 语义重排序（费用高，效果好）
    #   - bge: BGE-Reranker-v2-m3 本地重排序（免费，效果好，性能占用低）
    rerank_method: str = os.environ.get("RERANK_METHOD", "bge")
    reranker_type: str = os.environ.get("RERANKER_TYPE", "local")  # Reranker 类型: "api"(DashScope API), "local"(本地模型)
    bge_reranker_model: str = os.environ.get("BGE_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")  # BGE-Reranker 模型名

    # ── 查询改写策略 ───────────────────────────────────
    # QUERY_REWRITE_STRATEGY 可选值:
    #   - keyword: 基于关键词规则快速改写（无 LLM 调用，速度快）
    #   - llm: 使用 LLM 语义改写（更准确，需要 API 调用）
    query_rewrite_strategy: str = os.environ.get("QUERY_REWRITE_STRATEGY", "keyword")
    
    use_query_rewrite: bool = True  # 是否使用 Query 改写
    show_citations: bool = True  # 是否在 LLM 回答文本中标注信息来源（如【参考：xxx.pdf】】）
    show_retrieved_chunks: bool = True  # 是否在终端打印检索到的原始文档块内容（仅命令行模式生效）

    # ── 向量数据库配置（通过 VECTOR_DB_TYPE 切换）─────────────────
    # VECTOR_DB_TYPE 可选值:
    #   - faiss: 使用 FAISS（高性能，推荐）
    #   - memory: 使用内存实现（纯 Python，适合小规模数据）
    #   - auto: 自动选择（优先 FAISS，不可用时降级到内存）
    vector_db_type: str = os.environ.get("VECTOR_DB_TYPE", "auto")
    
    # ── BM25 + 向量混合检索配置 ─────────────────────────────────
    use_bm25: bool = True  # 是否启用 BM25 关键词检索
    bm25_k1: float = float(os.environ.get("BM25_K1", 1.5))  # BM25的k1参数，控制词频影响
    bm25_b: float = float(os.environ.get("BM25_B", 0.75))  # BM25的b参数，控制文档长度归一化
    bm25_weight: float = 0.5  # BM25 权重 (0~1)，0=纯向量检索，1=纯BM25，0.5=等权重
    rrf_k: int = int(os.environ.get("RRF_K", 60))  # RRF融合的K参数，控制不同排名结果的贡献
    rrf_candidate_factor: int = int(os.environ.get("RRF_CANDIDATE_FACTOR", 3))  # RRF候选数量因子（top_k的倍数）

    # ── 路径 ────────────────────────────────────────────────────
    # Path() 跨平台路径处理，比字符串拼接更健壮
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = project_root / "data"


# 模块级单例，其他地方 from config import config 即可使用
config = Config()
