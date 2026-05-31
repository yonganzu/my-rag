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

    # ── DashScope ──────────────────────────────────────────────
    # os.environ["KEY"] 会在缺失时抛 KeyError（及早暴露配置错误）
    # os.environ.get("KEY", "default") 有默认值，适合非敏感选项
    dashscope_api_key: str = os.environ["DASHSCOPE_API_KEY"]

    # ── LLM 配置 ───────────────────────────────────────────────
    llm_model: str = os.environ.get("LLM_MODEL", "qwen-plus")
    llm_base_url: str = os.environ.get(
        "LLM_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # ── Embedding 配置 ─────────────────────────────────────────
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "text-embedding-v4")

    # ── 文本分块配置 ────────────────────────────────────────────
    # chunk_size: 每块字符数（不是 token 数，但足够了演示）
    # chunk_overlap: 块与块之间的重叠字符数，避免切碎关键上下文
    chunk_size: int = 500
    chunk_overlap: int = 100

    # ── 检索配置 ────────────────────────────────────────────────
    top_k: int = 3  # 检索时返回最相似的 top-k 个块
    use_rerank: bool = True  # 是否使用语义重排序
    rerank_factor: int = 3    # rerank 时先获取的候选数量倍数
    use_query_rewrite: bool = True  # 是否使用 Query 改写

    # ── 路径 ────────────────────────────────────────────────────
    # Path() 跨平台路径处理，比字符串拼接更健壮
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = project_root / "data"


# 模块级单例，其他地方 from config import config 即可使用
config = Config()
