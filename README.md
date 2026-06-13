# 手搓 RAG 项目

从零实现的检索增强生成（Retrieval-Augmented Generation，RAG）系统，适合学习理解 RAG 的核心原理。

## 更新日志

### v1.2.0 (2026-06-13)
- **前后端完全分离**：`RAGPipeline` 注入 `ConversationManager`，后端自闭环上下文获取
- **查询改写模块独立**：新增 `src/query_rewriter.py`，支持 keyword/llm 两种改写策略
- **多轮对话上下文管理**：自动消解代词指代（如 "他和LLM的关系" → "RAG和LLM的关系"）
- **对话历史自闭环**：前端只需传 `user_id` + `conversation_id`，后端自动获取上下文
- **命令行会话管理独立**：新增 `src/cli_conversation.py`，从 `main.py` 解耦
- **Gradio 6.x 兼容**：修复 API 变更导致的前端组件兼容性问题
- **会话记录持久化修复**：修复首次消息不创建对话、助手回复不保存等 Bug
- **空对话过滤**：历史对话列表只显示有消息的对话

### v1.1.0 (2026-06-13)
- **权限控制**：多角色（admin/user）+ 文档级别硬控制
- **角色管理**：支持自定义角色和文档访问权限
- **BM25 去重**：修复重复文档导致词频偏差的问题
- **MemoryDB 修复**：修复单文档时 0 维数组索引错误

### v1.0.0 (2026-06-08)
- **模型配置化**：所有模型切换通过 `.env` 文件配置，支持 API/本地模式一键切换
- **BGE-Reranker-v2-m3 集成**：轻量级本地重排序模型，性能占用低，效果好
- **配置模板**：新增 `.env.example`，包含完整的配置说明
- **完全本地运行支持**：可配置为不依赖任何 API，纯本地运行
- **重排序方法扩展**：支持 `none`、`vector`、`keyword`、`llm`、`bge` 五种重排序方式
- **新增测试脚本**：`test_rerank.py`、`tests/test_bge_reranker.py`、`tests/test_qwen3_reranker.py`

### v0.9.0 (2026-06-08)
- **用户登录系统**：新增 `src/user_manager.py`，支持用户注册、登录、权限管理
- **历史对话功能**：新增 `src/conversation_manager.py`，支持多轮对话上下文管理
- **认证模块**：`src/auth.py` 提供统一的认证接口，支持 Session 管理
- **数据本地存储**：用户数据和对话历史存储在本地 JSON 文件
- **默认管理员**：首次运行自动创建 admin 账号（密码：admin）
- **新增文件**：`src/user_manager.py`、`src/conversation_manager.py`、`src/auth.py`

### v0.8.0 (2026-06-08)
- **本地 LLM 支持**：新增 `src/llm.py` 模块，支持 Ollama 和本地 Transformers 模型
- **本地 Embedding 支持**：`embedding.py` 支持 Sentence-BERT 系列本地模型
- **配置扩展**：新增 `LLM_TYPE` 和 `EMBEDDING_TYPE` 配置项，支持 `dashscope`、`ollama`、`local` 三种模式
- **懒加载机制**：本地模型采用懒加载，启动更快
- **新增依赖**：`sentence-transformers`、`ollama`

### v0.7.0 (2026-06-08)
- **BM25 检索独立**：将 BM25 从向量数据库中拆分出来，创建独立的 `BM25Retriever` 类
- **混合检索重构**：新增 `HybridRetriever` 类，专门负责向量 + BM25 的 RRF 融合
- **架构优化**：VectorDB 接口只负责纯向量检索，BM25 作为独立组件
- **代码复用**：BM25 实现只写一次，所有后端共享
- **易于测试**：可以单独测试 BM25 和向量检索的效果
- **移除旧文件**：删除了 `src/vector_store.py`，统一使用新的 `src/vector_db/` 模块

### v0.6.0 (2026-06-08)
- **向量数据库抽象层**：新增 `src/vector_db/` 目录，实现多后端支持
- **FAISS 集成**：高性能向量检索，支持百万级向量数据
- **自动降级机制**：FAISS 不可用时自动切换到内存实现
- **工厂模式**：`VectorDBFactory.create()` 支持 `faiss`、`memory`、`auto` 三种模式
- **接口统一**：所有后端实现相同的抽象接口，无缝切换
- **性能提升**：FAISS 比内存实现快 100 倍以上

### v0.5.0 (2026-06-07)
- **完整的评估体系**：新增 `tests/` 目录，包含全面的测试和评估工具
- **测试数据集**：从 15 个扩展至 68 个多样化的测试问题，覆盖 11 个类别
- **评估脚本**：`eval_to_file.py`、`simple_eval.py` 等用于测试系统性能
- **测试工具**：`test_env.py`、`test_retrieval.py`、`verify_test_data.py` 等辅助工具
- **规格文档**：新增 `SPEC.md`，定义评估基准、性能指标和后续改进计划
- **评估报告**：实现按类别统计分析，全面评估检索准确率和召回率
- **测试结果**：68 个问题，检索准确率和召回率均达到 100%

### v0.4.0 (2026-06-01)
- BM25 + 向量混合检索：用 RRF（倒数排名融合）算法融合关键词检索和语义检索
- 文档引用来源：回答中标注信息来源（文件名），可开关控制
- 代码重构：检索逻辑拆分到 `src/retrieval.py`，RAGPipeline 专注编排

### v0.3.0 (2026-05-31)
- ChatGPT 风格 Web 前端界面 (`app.py`)：左侧文档列表 + 右侧对话区域
- 文档更新检测功能：启动时对比文件元数据，自动识别新增/修改的文档
- 增量向量化：仅对变更文档重新向量化，大幅节省 API 调用量
- 文档列表实时展示：左侧栏显示已入库文档名和文件大小
- 向量库元数据持久化：新增 `metadata.json` 记录文档修改时间和大小
- `document_loader` 返回文档元数据，支持指定文件列表加载
- 修复 Embedding 模型默认值：`text-embedding-async-v1` → `text-embedding-v2`
- 前端功能：上传文档、清空知识库、清空对话

### v0.2.0 (2026-05-31)
- 增加了向量数据库持久化功能
- VectorStore 新增 `save()` 和 `load()` 方法
- RAGPipeline 会在构建知识库后自动保存到 `data/vector_db` 目录
- 支持通过 `load_knowledge_base()` 加载已有知识库
- 启动时自动检测本地向量数据库，避免重复调用 API
- Embedding 分批处理，支持大批量文档（API 限制每批最多 10 条）
- 回答时显示检索到的相关文档块（上下文来源）
- 增加对多种文档格式的支持：Word (.docx)、Excel (.xlsx)、PowerPoint (.pptx)、PDF、HTML
- 增加语义重排序（Rerank）功能，通过 LLM 对检索结果进行二次排序，提升相关性
- 增加 Query 改写功能，通过 LLM 优化用户查询，提升检索召回率

### v0.1.0 (2026-05-29)
- 初始版本，包含完整的 RAG 链路实现

## 项目简介

本项目不使用任何现成的 RAG 框架（如 LangChain、LlamaIndex），而是从零手动实现了完整的 RAG 链路，帮助你理解：
- 文档加载与分块
- 文本向量化（Embedding）
- 向量存储与检索
- BM25 关键词检索
- 混合检索（向量 + BM25）
- 提示词构建
- LLM 调用与回答生成
- Query 改写与 Rerank 重排序

## 技术栈

- **Python 3.13+**
- **Gradio 6**（Web 前端界面）
- **阿里云 DashScope API**（用于 Embedding 和 LLM）
- **NumPy**（用于向量计算）
- **jieba**（中文分词，用于 BM25）
- **python-dotenv**（环境变量管理）

## 项目结构

```
learn/
├── .gitignore
├── .env.example            # 配置模板
├── README.md
├── SPEC.md                 # 规格文档
├── requirements.txt
├── pyproject.toml
├── start.ps1               # Windows 一键启动
├── main.py                 # 命令行入口
├── app.py                  # Web 前端入口 (Gradio 6)
├── data/
│   ├── documents/          # 文档存放目录
│   │   └── sample_document.txt
│   ├── vector_db/          # 向量库持久化目录（运行时生成）
│   ├── users.json          # 用户数据（运行时生成）
│   ├── roles.json          # 角色权限配置（运行时生成）
│   └── conversations/      # 对话历史（运行时生成）
├── src/
│   ├── __init__.py
│   ├── config.py           # 配置管理（支持 .env 覆盖）
│   ├── document_loader.py  # 文档加载与分块
│   ├── embedding.py        # 文本向量化
│   ├── llm.py              # LLM 接口（支持 DashScope/Ollama/Local）
│   ├── retrieval.py        # 检索器（混合检索/Query改写/Rerank）
│   ├── rag_pipeline.py     # RAG 流水线（编排 + 生成 + 上下文管理）
│   ├── auth.py             # 认证模块
│   ├── user_manager.py     # 用户管理
│   ├── conversation_manager.py  # 对话管理与长期记忆
│   ├── cli_conversation.py # 命令行会话管理UI
│   ├── query_rewriter.py   # 查询改写（代词消解，keyword/llm 双策略）
│   └── vector_db/          # 向量数据库模块
│       ├── __init__.py
│       ├── base.py         # 抽象接口定义
│       ├── faiss_db.py     # FAISS 实现（高性能）
│       ├── memory_db.py    # 内存实现（降级方案）
│       ├── bm25_retriever.py    # BM25 关键词检索器
│       └── hybrid_retriever.py  # 混合检索器（向量 + BM25）
└── tests/                  # 测试与评估目录
    ├── test_retrieval.py
    ├── test_bge_reranker.py
    ├── test_qwen3_reranker.py
    ├── test_conversation.py
    ├── test_permission_filter.py
    ├── test_rag_pipeline_metadata.py
    └── eval_to_file.py
```

## 核心模块详解

### 1. config.py - 配置管理

使用 `dataclass` 定义类型化的配置，通过环境变量加载 API Key，敏感信息不硬编码。

### 2. document_loader.py - 文档加载与分块

支持格式：`.txt`、`.docx`、`.xlsx`、`.pptx`、`.pdf`、`.html`

返回：(chunks, doc_metadata, chunk_sources) — 文本块、元数据、来源文件名

### 3. embedding.py - 文本向量化

将文本转换为向量，支持两种模式：
- **DashScope API**：使用阿里云的 `text-embedding-v2` 模型
- **本地模型**：使用 Sentence-BERT 系列模型（如 `all-MiniLM-L6-v2`）

### 4. llm.py - 大语言模型接口

统一的 LLM 调用接口，支持三种模式：
- **DashScope API**：调用通义千问等云端模型
- **Ollama**：本地运行开源模型（如 Qwen、Llama3）
- **Transformers**：直接使用 HuggingFace 模型

### 5. src/vector_db/ - 向量数据库模块

统一的向量数据库抽象层，支持多后端：

| 文件 | 功能 |
|------|------|
| `base.py` | 抽象基类，定义统一接口 |
| `memory_db.py` | 内存向量存储实现 |
| `faiss_db.py` | FAISS 高性能向量检索 |
| `bm25_retriever.py` | 独立的 BM25 关键词检索器 |
| `hybrid_retriever.py` | 混合检索器，用 RRF 融合向量+BM25 |

### 6. retrieval.py - 检索器

负责检索全流程：
- Query 改写：用 LLM 优化模糊问题
- 混合检索：向量 + BM25 互补
- Rerank 重排序：用 LLM 对候选结果重新排序

### 7. rag_pipeline.py - RAG 流水线

串联所有模块，专注编排。注入 `ConversationManager` 后后端自闭环：

```
前端只传: question + user_id + conversation_id
    │
    ↓
┌─────────────────────┐
│  获取对话上下文      │  后端自动从 ConversationManager 获取历史
│  _get_conversation_context()
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│  Query 改写          │  query_rewriter 解决代词指代
│  rewrite_query()     │  "他" → "RAG"
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│  Retrieval           │  Embedding → 混合检索 → Rerank
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│  构建 Prompt         │  改写后的问题 + 检索上下文 + 对话历史
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│  LLM 生成            │  通过 llm.py 调用模型生成最终回答
└─────────────────────┘
```

### 8. auth.py - 认证模块

统一的认证接口，整合用户管理和会话管理：

- **登录/登出**：基于 Session 的认证
- **密码安全**：PBKDF2 哈希 + 随机盐值
- **会话管理**：24 小时自动过期，支持刷新

### 9. user_manager.py - 用户管理

用户数据管理，支持多用户：

- **用户注册/登录**：安全的密码存储
- **角色管理**：支持 admin 和 user 两种角色
- **数据存储**：本地 JSON 文件

### 10. conversation_manager.py - 对话管理与长期记忆

对话历史管理，支持多轮上下文：

- **多轮对话**：保存完整的对话历史
- **上下文修剪**：保留最近 N 条消息，控制 token 消耗
- **上下文压缩**：将早期消息压缩为摘要
- **长期记忆**：自动提取用户偏好和重要信息
- **用户隔离**：每个用户只能访问自己的对话

### 11. query_rewriter.py - 查询改写

独立的查询改写模块，解决多轮对话中的代词指代问题：

- **keyword 策略**：基于正则规则从对话历史提取实体，无 LLM 调用，极快
- **llm 策略**：使用 LLM 语义理解改写，更准确，需要 API 调用
- **策略可切换**：通过 `QUERY_REWRITE_STRATEGY` 环境变量控制

改写示例：`"他和LLM的关系"` → `"RAG和LLM的关系"`

### 12. cli_conversation.py - 命令行会话管理

命令行版会话管理 UI，从 `main.py` 解耦：

- **会话创建/切换/删除**：完整的命令行交互
- **会话列表**：显示所有历史对话
- **代码高内聚**：所有会话管理逻辑集中在一个模块

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境（Windows）
.venv\Scripts\activate.bat

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

在项目根目录创建 `.env` 文件：

```env
DASHSCOPE_API_KEY=sk-你的APIKey
LLM_MODEL=qwen-plus
EMBEDDING_MODEL=text-embedding-v2
```

获取 API Key：https://dashscope.console.aliyun.com/

### 3. 运行项目

**Web 界面（推荐）：**

```bash
python app.py
```

然后在浏览器中访问 `http://localhost:7860`，即可使用 ChatGPT 风格的前端界面。

**命令行模式：**

```bash
python main.py
```

或者在 Windows 上直接双击 `run.bat`。

## 功能配置

### 通过 .env 文件配置（推荐）

复制 `.env.example` 为 `.env` 并修改配置：

```env
# API Key（必填，使用 API 模式时）
DASHSCOPE_API_KEY=sk-你的APIKey

# LLM 配置
LLM_TYPE=dashscope          # dashscope (API) / ollama (本地) / local (本地 Transformers)
LLM_MODEL=qwen-plus

# Embedding 配置
EMBEDDING_TYPE=dashscope    # dashscope (API) / local (本地模型)
EMBEDDING_MODEL=text-embedding-v2
EMBEDDING_LOCAL_MODEL=all-MiniLM-L6-v2

# 重排序配置
RERANK_METHOD=bge           # none / vector / keyword / llm / bge
RERANKER_TYPE=local         # api / local
BGE_RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# 向量数据库配置
VECTOR_DB_TYPE=auto         # faiss / memory / auto
```

### 配置项说明

| 配置项 | 可选值 | 说明 |
|--------|--------|------|
| `LLM_TYPE` | `dashscope`, `ollama`, `local` | LLM 类型切换 |
| `EMBEDDING_TYPE` | `dashscope`, `local` | Embedding 类型切换 |
| `RERANK_METHOD` | `none`, `vector`, `keyword`, `llm`, `bge` | 重排序方法切换 |
| `VECTOR_DB_TYPE` | `faiss`, `memory`, `auto` | 向量数据库切换 |
| `QUERY_REWRITE_STRATEGY` | `keyword`, `llm` | 查询改写策略（代词消解） |

### 完全本地运行配置

```env
LLM_TYPE=ollama
LLM_MODEL=qwen2.5:7b
EMBEDDING_TYPE=local
EMBEDDING_LOCAL_MODEL=all-MiniLM-L6-v2
RERANK_METHOD=bge
RERANKER_TYPE=local
VECTOR_DB_TYPE=faiss
```

### 在 src/config.py 中配置（高级选项）

```python
# 检索配置
top_k: int = 3                    # 检索返回的文档块数量
use_bm25: bool = True             # 是否启用 BM25 混合检索
bm25_weight: float = 0.5          # BM25 权重（0=纯向量，1=纯BM25）
use_rerank: bool = True           # 是否启用重排序
use_query_rewrite: bool = True    # 是否启用 Query 改写
show_citations: bool = True       # 是否显示引用来源
show_retrieved_chunks: bool = True # 是否显示检索到的文档块

# BM25 参数
bm25_k1: float = 1.5              # BM25 的 k1 参数
bm25_b: float = 0.75              # BM25 的 b 参数

# RRF 融合参数
rrf_k: int = 60                   # RRF 融合的 K 参数
rrf_candidate_factor: int = 3     # RRF 候选数量因子
```

## RAG 原理

### 什么是 RAG？

RAG（检索增强生成）是一种将信息检索与文本生成结合的技术。

### 为什么需要 RAG？

| 纯 LLM | RAG |
|--------|-----|
| 知识有截止日期 | 可以访问最新数据 |
| 不知道私有数据 | 可以检索私有知识库 |
| 容易产生幻觉 | 回答基于实际检索内容，更可信 |
| 知识库更新需重新训练 | 知识库可随时更新 |

### 混合检索原理

**BM25**：基于词频和逆文档频率的传统关键词检索，擅长精确匹配术语。

**向量检索**：将文本转为语义向量，用余弦相似度匹配，擅长理解语义相近但用词不同的问题。

**RRF 融合**：将两种检索的结果按排名融合，取长补短。

```
用户问题 ──→ 向量检索 ──→ 排名①
     │                      ↓
     │              ┌────── RRF ──→ 最终排序
     │              ↑
     ↓              │
  BM25 检索 ──→ 排名②
```

## 扩展方向

1. **多模态支持**：添加图片、视频等非文本内容的处理
2. **Docker 部署**：封装为 Docker 镜像便于分发与部署
3. **并发安全**：JSON 文件读写加锁，支持多用户同时操作
4. **评估指标体系**：添加 RAGAS 等自动化评估框架
5. **API 服务化**：将核心功能封装为 RESTful API 供外部调用

## 许可证

MIT License，随便玩！

## 交流

有问题或想法？欢迎讨论！

---

**祝你学习愉快！** 🚀
