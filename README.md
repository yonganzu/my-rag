# 手搓 RAG 项目

从零实现的检索增强生成（Retrieval-Augmented Generation，RAG）系统，适合学习理解 RAG 的核心原理。

## 更新日志

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

## 📚 项目简介

本项目不使用任何现成的 RAG 框架（如 LangChain、LlamaIndex），而是从零手动实现了完整的 RAG 链路，帮助你理解：
- 文档加载与分块
- 文本向量化（Embedding）
- 向量存储与检索
- 提示词构建
- LLM 调用与回答生成

## 🛠️ 技术栈

- **Python 3.13+**
- **Gradio 6**（Web 前端界面）
- **阿里云 DashScope API**（用于 Embedding 和 LLM）
- **NumPy**（用于向量计算）
- **jieba**（中文分词）
- **python-dotenv**（环境变量管理）

## 📂 项目结构

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
└── src/
    ├── __init__.py
    ├── config.py        # 配置管理
    ├── document_loader.py  # 文档加载与分块
    ├── embedding.py     # 文本向量化
    ├── vector_store.py  # 向量存储与检索
    └── rag_pipeline.py  # RAG 流水线
```

## 🔍 核心模块详解

### 1. config.py - 配置管理

使用 `dataclass` 定义类型化的配置，通过环境变量加载 API Key，敏感信息不硬编码。

### 2. document_loader.py - 文档加载与分块

**为什么要分块？**
- LLM 有上下文窗口限制
- 检索时细粒度的块更匹配问题
- 块之间保持重叠，避免关键信息被截断

**分块策略：**
- 固定字符数作为基础
- 优雅切分：尝试在句号、换行等句子边界断开

### 3. embedding.py - 文本向量化

将文本转换为向量（Embedding），使用阿里云 DashScope 的 `text-embedding-v4` 模型。

**Embedding 的作用：**
- 将非结构化文本转换为计算机可计算的向量
- 语义相关的文本在向量空间中距离更近
- 支持通过余弦相似度计算文本相似度

### 4. vector_store.py - 向量存储与检索

一个简单的内存向量存储实现，使用余弦相似度检索。

**余弦相似度：**
```
cosine(A, B) = (A · B) / (||A|| * ||B||)
```
范围 [-1, 1]，越接近 1 表示越相似。

### 5. rag_pipeline.py - RAG 流水线

整个 RAG 系统的核心，串联所有模块：

```
用户问题
    ↓
问题向量化
    ↓
向量检索（找到最相关的文档块）
    ↓
构建 Prompt（问题 + 检索到的上下文）
    ↓
LLM 生成回答
```

## 🚀 快速开始

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

## 📖 RAG 原理

### 什么是 RAG？

RAG（检索增强生成）是一种将信息检索与文本生成结合的技术。

### 为什么需要 RAG？

| 纯 LLM | RAG |
|--------|-----|
| 知识有截止日期 | 可以访问最新数据 |
| 不知道私有数据 | 可以检索私有知识库 |
| 容易产生幻觉 | 回答基于实际检索内容，更可信 |
| 知识库更新需重新训练 | 知识库可随时更新 |

### RAG 的工作流程

```
┌─────────────────────────────────────────────────────────┐
│                     1. 知识库构建                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  文档 → 分块 → 向量化 → 向量存储                          │
│                                                          │
└─────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────┐
│                      2. 问答流程                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  用户问题 → 向量化 → 检索 → 上下文 + 问题 → LLM → 回答  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## 🎯 示例对话

```
📄 加载文档: data\sample_document.txt

✅ 文档被分割为 3 个文本块

正在对 3 个文档块生成向量嵌入...
知识库构建完成，共 3 个文档块

============================================================
💡 RAG 问答系统已就绪！（输入 'exit' 退出）
============================================================

❓ 请输入问题: 什么是 Transformer？
🔍 正在检索和生成回答...
检索到 3 个相关文档块

🤖 回答:
Transformer 是 Google 在 2017 年提出的几乎所有现代大语言模型的基础架构。它的核心创新是自注意力机制（Self-Attention），允许模型在处理每个词时"关注"句子中所有其他词的位置，从而捕捉长距离依赖关系。与之前的 RNN 或 LSTM 不同，Transformer 可以并行处理整个序列，大大提高了训练效率。Transformer 由编码器（Encoder）和解码器（Decoder）两部分组成。BERT 只使用编码器部分，GPT 只使用解码器部分。
```

## 🔄 扩展方向

如果你想进一步扩展这个项目，可以考虑：

1. **更换向量数据库**：用 FAISS、Chroma 或 Milvus 替换当前的内存存储
2. **添加重排序（Reranker）**：在检索后使用更好的模型对结果重新排序
3. **支持多模态**：添加图片、视频的处理
4. **Docker 部署**：封装为 Docker 镜像便于分发
5. **添加用户管理**：多用户支持和对话历史保存

## 📄 许可证

MIT License，随便玩！

## 📬 交流

有问题或想法？欢迎讨论！

---

**祝你学习愉快！** 🚀
