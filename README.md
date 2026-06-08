# 手搓 RAG 项目

从零实现的检索增强生成（Retrieval-Augmented Generation，RAG）系统，适合学习理解 RAG 的核心原理。

## 更新日志

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
├── README.md
├── requirements.txt
├── pyproject.toml
├── run.bat              # Windows 命令行启动
├── run_app.bat          # Windows Web 界面启动
├── main.py              # 命令行入口
├── app.py               # Web 前端入口 (Gradio)
├── data/
│   ├── documents/       # 文档存放目录
│   │   └── sample_document.txt
│   └── vector_db/       # 向量库持久化目录（运行时生成）
├── src/
│   ├── __init__.py
│   ├── config.py        # 配置管理
│   ├── document_loader.py  # 文档加载与分块
│   ├── embedding.py     # 文本向量化
│   ├── retrieval.py     # 检索器（混合检索/Query改写/Rerank）
│   ├── rag_pipeline.py  # RAG 流水线（编排 + 生成）
│   └── vector_db/       # 向量数据库模块
│       ├── __init__.py
│       ├── base.py      # 抽象接口定义
│       ├── faiss_db.py  # FAISS 实现（高性能）
│       ├── memory_db.py # 内存实现（降级方案）
│       ├── bm25_retriever.py   # BM25 关键词检索器
│       └── hybrid_retriever.py # 混合检索器（向量 + BM25）
└── tests/               # 测试与评估目录
```

## 核心模块详解

### 1. config.py - 配置管理

使用 `dataclass` 定义类型化的配置，通过环境变量加载 API Key，敏感信息不硬编码。

### 2. document_loader.py - 文档加载与分块

支持格式：`.txt`、`.docx`、`.xlsx`、`.pptx`、`.pdf`、`.html`

返回：(chunks, doc_metadata, chunk_sources) — 文本块、元数据、来源文件名

### 3. embedding.py - 文本向量化

将文本转换为向量，使用阿里云 DashScope 的 `text-embedding-v2` 模型。

### 4. vector_store.py - 向量存储与 BM25 索引

一个内存向量存储实现，包含：
- **向量检索**：余弦相似度计算
- **BM25 索引**：基于词频的关键词检索
- **混合检索**：`hybrid_search()` 用 RRF 算法融合两者

### 5. retrieval.py - 检索器

负责检索全流程：
- Query 改写：用 LLM 优化模糊问题
- 混合检索：向量 + BM25 互补
- Rerank 重排序：用 LLM 对候选结果重新排序

### 6. rag_pipeline.py - RAG 流水线

串联所有模块，专注编排：

```
用户问题
    ↓
┌─────────────┐
│  Retrieval   │  Query改写 → Embedding → 混合检索 → Rerank
└──────┬──────┘
       ↓
┌─────────────┐
│  构建 Prompt │  问题 + 检索到的上下文 + 引用来源
└──────┬──────┘
       ↓
┌─────────────┐
│  LLM 生成   │  调用通义千问生成最终回答
└─────────────┘
```

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

在 `src/config.py` 中可以开关各项功能：

```python
# 检索配置
top_k: int = 3                    # 检索返回的文档块数量
use_bm25: bool = True             # 是否启用 BM25 混合检索
bm25_weight: float = 0.5          # BM25 权重（0=纯向量，1=纯BM25）
use_rerank: bool = True           # 是否启用 LLM 重排序
use_query_rewrite: bool = True    # 是否启用 Query 改写
show_citations: bool = True       # 是否显示引用来源
show_retrieved_chunks: bool = True # 是否显示检索到的文档块
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

如果你想进一步扩展这个项目，可以考虑：

1. **更换向量数据库**：用 FAISS、Chroma 或 Milvus 替换当前的内存存储
2. **支持多模态**：添加图片、视频的处理
3. **Docker 部署**：封装为 Docker 镜像便于分发
4. **添加用户管理**：多用户支持和对话历史保存
5. **评估指标**：添加召回率、准确率等评估功能

## 许可证

MIT License，随便玩！

## 交流

有问题或想法？欢迎讨论！

---

**祝你学习愉快！** 🚀
