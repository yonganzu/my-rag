"""
测试数据集：用于评估 RAG 系统的检索和生成性能

数据格式：
- question: 用户问题
- expected_answer: 期望的标准答案（可用于评估生成质量）
- expected_sources: 期望的来源文档（用于评估检索召回率）
- category: 问题类别（用于分组分析）
"""

test_dataset = [
    # 原有测试问题（15个）
    {
        "question": "什么是 Transformer 架构？",
        "expected_answer": "Transformer 是几乎所有现代大语言模型的基础架构，由 Google 在 2017 年提出。它的核心创新是自注意力机制（Self-Attention），允许模型在处理每个词时关注句子中所有其他词的位置，从而捕捉长距离依赖关系。",
        "expected_sources": ["sample_document.txt"],
        "category": "概念定义"
    },
    {
        "question": "Transformer 的核心创新是什么？",
        "expected_answer": "Transformer 的核心创新是自注意力机制（Self-Attention），允许模型在处理每个词时关注句子中所有其他词的位置，从而捕捉长距离依赖关系。",
        "expected_sources": ["sample_document.txt"],
        "category": "核心技术"
    },
    {
        "question": "BERT 和 GPT 有什么区别？",
        "expected_answer": "BERT 只使用 Transformer 的编码器部分，GPT 只使用解码器部分。",
        "expected_sources": ["sample_document.txt"],
        "category": "技术对比"
    },
    {
        "question": "大语言模型的训练分为哪几个阶段？",
        "expected_answer": "大语言模型的训练通常分为三个阶段：预训练（Pre-training）、有监督微调（Supervised Fine-Tuning, SFT）和强化学习人类反馈（RLHF）阶段。",
        "expected_sources": ["sample_document.txt"],
        "category": "训练流程"
    },
    {
        "question": "什么是 RAG？",
        "expected_answer": "RAG（Retrieval-Augmented Generation）是一种将信息检索与文本生成结合的技术架构，先在外部知识库中检索与用户问题相关的信息片段，然后将这些信息作为上下文提供给 LLM 生成回答。",
        "expected_sources": ["sample_document.txt"],
        "category": "概念定义"
    },
    {
        "question": "RAG 有哪些优势？",
        "expected_answer": "RAG 相比纯 LLM 方案有三个显著优势：可以访问私有或最新的数据、有效减少幻觉、知识库可以随时更新无需重新训练模型。",
        "expected_sources": ["sample_document.txt"],
        "category": "优势特点"
    },
    {
        "question": "RAG 系统包含哪些核心环节？",
        "expected_answer": "RAG 系统通常包含文档加载与分块、向量化与索引、检索与生成四个核心环节。",
        "expected_sources": ["sample_document.txt"],
        "category": "系统架构"
    },
    {
        "question": "常用的向量数据库有哪些？",
        "expected_answer": "常用的向量数据库包括 FAISS、Chroma、Milvus、Pinecone。",
        "expected_sources": ["sample_document.txt"],
        "category": "工具推荐"
    },
    {
        "question": "什么是提示工程？",
        "expected_answer": "提示工程是优化输入提示词以获得理想输出的一门技术，常见的技巧包括少样本提示、思维链、角色提示等。",
        "expected_sources": ["sample_document.txt"],
        "category": "概念定义"
    },
    {
        "question": "什么是 Agent 智能体？",
        "expected_answer": "Agent 是指能够自主使用工具、规划和执行任务的 AI 系统，工作流程包括接收指令、理解意图、规划步骤、调用工具、综合结果和返回回答。",
        "expected_sources": ["sample_document.txt"],
        "category": "概念定义"
    },
    {
        "question": "选择向量数据库需要考虑哪些因素？",
        "expected_answer": "选择向量数据库时需要考虑数据规模、查询延迟、运维成本和一致性要求。",
        "expected_sources": ["sample_document.txt"],
        "category": "选型建议"
    },
    {
        "question": "RLHF 是什么？",
        "expected_answer": "RLHF 是强化学习人类反馈（Reinforcement Learning from Human Feedback）的缩写，是大语言模型训练的第三阶段，通过人类偏好进一步优化模型的输出质量。",
        "expected_sources": ["sample_document.txt"],
        "category": "概念定义"
    },
    {
        "question": "什么是思维链（Chain-of-Thought）？",
        "expected_answer": "思维链（Chain-of-Thought, CoT）是一种提示工程技巧，引导模型逐步推理以获得更好的输出。",
        "expected_sources": ["sample_document.txt"],
        "category": "提示工程"
    },
    {
        "question": "FAISS 是什么？",
        "expected_answer": "FAISS 是 Facebook 开源的向量数据库，适合小规模本地部署。",
        "expected_sources": ["sample_document.txt"],
        "category": "工具介绍"
    },
    {
        "question": "什么是自监督学习？",
        "expected_answer": "自监督学习是一种机器学习范式，在预训练阶段模型在海量的无标注文本上进行学习，目标是预测下一个 token，从而赋予模型广泛的语言知识和世界知识。",
        "expected_sources": ["sample_document.txt"],
        "category": "概念定义"
    },

    # 新增测试问题（45个，共60个）
    {
        "question": "Transformer 和之前的 RNN、LSTM 有什么区别？",
        "expected_answer": "Transformer 可以并行处理整个序列，大大提高了训练效率，而 RNN 和 LSTM 只能顺序处理。",
        "expected_sources": ["sample_document.txt"],
        "category": "技术对比"
    },
    {
        "question": "预训练阶段的目标是什么？",
        "expected_answer": "预训练阶段的目标是预测下一个 token，让模型在海量的无标注文本上进行自监督学习，从而获得广泛的语言知识和世界知识。",
        "expected_sources": ["sample_document.txt"],
        "category": "训练流程"
    },
    {
        "question": "有监督微调（SFT）阶段做什么？",
        "expected_answer": "有监督微调（SFT）阶段使用人工标注的高质量问答数据对模型进行训练，让模型学会遵循指令。",
        "expected_sources": ["sample_document.txt"],
        "category": "训练流程"
    },
    {
        "question": "强化学习人类反馈（RLHF）阶段的作用是什么？",
        "expected_answer": "RLHF 阶段通过人类偏好进一步优化模型的输出质量。",
        "expected_sources": ["sample_document.txt"],
        "category": "训练流程"
    },
    {
        "question": "哪些商业模型采用了预训练、SFT、RLHF 这样的训练范式？",
        "expected_answer": "GPT-4、Claude、通义千问等商业模型都采用了预训练、SFT、RLHF 类似的训练范式。",
        "expected_sources": ["sample_document.txt"],
        "category": "实际应用"
    },
    {
        "question": "提示工程中的少样本提示是什么？",
        "expected_answer": "少样本提示（Few-shot）是指在提示中给出几个示例，帮助模型理解任务要求。",
        "expected_sources": ["sample_document.txt"],
        "category": "提示工程"
    },
    {
        "question": "角色提示有什么作用？",
        "expected_answer": "角色提示（Role Prompting）是指给模型一个身份设定，帮助模型更好地完成特定任务。",
        "expected_sources": ["sample_document.txt"],
        "category": "提示工程"
    },
    {
        "question": "好的提示词有什么作用？",
        "expected_answer": "好的提示词可以显著提升模型在特定任务上的表现。",
        "expected_sources": ["sample_document.txt"],
        "category": "提示工程"
    },
    {
        "question": "RAG 系统如何减少幻觉？",
        "expected_answer": "RAG 系统的回答基于检索到的实际内容，因此可以有效减少幻觉。",
        "expected_sources": ["sample_document.txt"],
        "category": "优势特点"
    },
    {
        "question": "RAG 系统的知识库可以随时更新吗？",
        "expected_answer": "是的，RAG 系统的知识库可以随时更新，无需重新训练模型。",
        "expected_sources": ["sample_document.txt"],
        "category": "优势特点"
    },
    {
        "question": "RAG 能否访问私有或最新的数据？",
        "expected_answer": "是的，RAG 相比纯 LLM 方案可以访问私有或最新的数据。",
        "expected_sources": ["sample_document.txt"],
        "category": "优势特点"
    },
    {
        "question": "向量数据库的作用是什么？",
        "expected_answer": "向量数据库专门用于存储和检索高维向量。在 RAG 系统中，文档被转换成向量后存入向量数据库，查询时计算问题向量与文档向量的相似度。",
        "expected_sources": ["sample_document.txt"],
        "category": "系统架构"
    },
    {
        "question": "Chroma 适合什么场景？",
        "expected_answer": "Chroma 是轻量级的向量数据库，适合原型开发。",
        "expected_sources": ["sample_document.txt"],
        "category": "工具推荐"
    },
    {
        "question": "Milvus 适合什么场景？",
        "expected_answer": "Milvus 是分布式向量数据库，适合大规模生产环境。",
        "expected_sources": ["sample_document.txt"],
        "category": "工具推荐"
    },
    {
        "question": "Pinecone 有什么特点？",
        "expected_answer": "Pinecone 是全托管云服务向量数据库。",
        "expected_sources": ["sample_document.txt"],
        "category": "工具推荐"
    },
    {
        "question": "Agent 系统的工作流程是什么？",
        "expected_answer": "Agent 的典型工作流程是：接收用户指令 → 理解意图 → 规划步骤 → 调用工具（如搜索、计算器、API）→ 综合结果 → 返回回答。",
        "expected_sources": ["sample_document.txt"],
        "category": "系统架构"
    },
    {
        "question": "Agent 系统的关键能力有哪些？",
        "expected_answer": "Agent 系统的关键能力包括工具调用（Function Calling）、任务规划和记忆管理。",
        "expected_sources": ["sample_document.txt"],
        "category": "核心技术"
    },
    {
        "question": "当前流行的 Agent 框架有哪些？",
        "expected_answer": "LangChain 和 AutoGPT 是当前流行的 Agent 框架。",
        "expected_sources": ["sample_document.txt"],
        "category": "工具推荐"
    },
    {
        "question": "Agent 被认为是通往什么的重要方向？",
        "expected_answer": "Agent 被认为是通往通用人工智能的重要方向之一。",
        "expected_sources": ["sample_document.txt"],
        "category": "实际应用"
    },
    {
        "question": "自注意力机制有什么作用？",
        "expected_answer": "自注意力机制（Self-Attention）允许模型在处理每个词时关注句子中所有其他词的位置，从而捕捉长距离依赖关系。",
        "expected_sources": ["sample_document.txt"],
        "category": "核心技术"
    },
    {
        "question": "Transformer 由哪两部分组成？",
        "expected_answer": "Transformer 由编码器（Encoder）和解码器（Decoder）两部分组成。",
        "expected_sources": ["sample_document.txt"],
        "category": "系统架构"
    },
    {
        "question": "哪年提出的 Transformer？",
        "expected_answer": "Transformer 由 Google 在 2017 年提出。",
        "expected_sources": ["sample_document.txt"],
        "category": "概念定义"
    },
    {
        "question": "RAG 为什么要先检索相关信息片段？",
        "expected_answer": "RAG 先在外部知识库中检索与用户问题相关的信息片段，然后将这些信息作为上下文提供给 LLM 生成回答，以确保回答的准确性和时效性。",
        "expected_sources": ["sample_document.txt"],
        "category": "系统架构"
    },
    {
        "question": "向量数据库如何工作？",
        "expected_answer": "在 RAG 系统中，文档被转换成向量后存入向量数据库，查询时计算问题向量与文档向量的相似度来检索相关文档。",
        "expected_sources": ["sample_document.txt"],
        "category": "系统架构"
    },
    {
        "question": "提示工程的常见技巧有哪些？",
        "expected_answer": "提示工程的常见技巧包括：少样本提示（Few-shot）、思维链（Chain-of-Thought, CoT）、角色提示（Role Prompting）。",
        "expected_sources": ["sample_document.txt"],
        "category": "提示工程"
    },
    {
        "question": "大语言模型预训练阶段学什么？",
        "expected_answer": "大语言模型在预训练阶段在海量的无标注文本上进行自监督学习，目标是预测下一个 token，这个阶段赋予了模型广泛的语言知识和世界知识。",
        "expected_sources": ["sample_document.txt"],
        "category": "训练流程"
    },
    {
        "question": "选择向量数据库时，数据规模是考虑因素吗？",
        "expected_answer": "是的，选择向量数据库时需要考虑数据规模、查询延迟、运维成本和一致性要求等因素。",
        "expected_sources": ["sample_document.txt"],
        "category": "选型建议"
    },
    {
        "question": "Agent 需要调用工具吗？",
        "expected_answer": "是的，Agent 是能够自主使用工具的 AI 系统，工作流程中包括调用工具（如搜索、计算器、API）这一步骤。",
        "expected_sources": ["sample_document.txt"],
        "category": "核心技术"
    },
    {
        "question": "纯 LLM 方案和 RAG 方案有什么区别？",
        "expected_answer": "纯 LLM 方案没有外部知识库检索步骤，而 RAG 方案先在外部知识库中检索相关信息片段，然后将这些信息作为上下文提供给 LLM 生成回答。RAG 相比纯 LLM 可以访问私有或最新的数据、有效减少幻觉、知识库可以随时更新无需重新训练模型。",
        "expected_sources": ["sample_document.txt"],
        "category": "技术对比"
    },
    {
        "question": "提示工程是一门什么技术？",
        "expected_answer": "提示工程是优化输入提示词以获得理想输出的一门技术。",
        "expected_sources": ["sample_document.txt"],
        "category": "概念定义"
    },
    {
        "question": "思维链（Chain-of-Thought）有什么作用？",
        "expected_answer": "思维链（Chain-of-Thought, CoT）是一种提示工程技巧，引导模型逐步推理以获得更好的输出。",
        "expected_sources": ["sample_document.txt"],
        "category": "提示工程"
    },
    {
        "question": "Transformer 的并行处理有什么优势？",
        "expected_answer": "Transformer 可以并行处理整个序列，大大提高了训练效率，这是与之前的 RNN 或 LSTM 相比的重要优势。",
        "expected_sources": ["sample_document.txt"],
        "category": "优势特点"
    },
    {
        "question": "RAG 系统如何更新知识库？",
        "expected_answer": "RAG 系统的知识库可以随时更新，无需重新训练模型，这是 RAG 相比纯 LLM 的重要优势之一。",
        "expected_sources": ["sample_document.txt"],
        "category": "系统架构"
    },
    {
        "question": "向量数据库中的文档是如何存储的？",
        "expected_answer": "在 RAG 系统中，文档被转换成向量后存入向量数据库。",
        "expected_sources": ["sample_document.txt"],
        "category": "系统架构"
    },
    {
        "question": "查询时向量数据库做什么？",
        "expected_answer": "查询时，向量数据库计算问题向量与文档向量的相似度来检索相关文档。",
        "expected_sources": ["sample_document.txt"],
        "category": "系统架构"
    },
    {
        "question": "谁开源了 FAISS？",
        "expected_answer": "FAISS 是 Facebook 开源的向量数据库。",
        "expected_sources": ["sample_document.txt"],
        "category": "工具介绍"
    },
    {
        "question": "有监督微调（SFT）使用什么数据？",
        "expected_answer": "有监督微调（SFT）阶段使用人工标注的高质量问答数据对模型进行训练，让模型学会遵循指令。",
        "expected_sources": ["sample_document.txt"],
        "category": "训练流程"
    },
    {
        "question": "预训练阶段的数据标注情况是？",
        "expected_answer": "预训练阶段模型在海量的无标注文本上进行自监督学习，不需要人工标注数据。",
        "expected_sources": ["sample_document.txt"],
        "category": "训练流程"
    },
    {
        "question": "Agent 系统的工具调用能力指的是？",
        "expected_answer": "Agent 系统的关键能力包括工具调用（Function Calling）、任务规划和记忆管理。工具调用指 Agent 可以自主使用搜索、计算器、API 等工具。",
        "expected_sources": ["sample_document.txt"],
        "category": "核心技术"
    },
    {
        "question": "BERT 只使用 Transformer 的哪个部分？",
        "expected_answer": "BERT 只使用 Transformer 的编码器（Encoder）部分。",
        "expected_sources": ["sample_document.txt"],
        "category": "技术对比"
    },
    {
        "question": "GPT 只使用 Transformer 的哪个部分？",
        "expected_answer": "GPT 只使用 Transformer 的解码器（Decoder）部分。",
        "expected_sources": ["sample_document.txt"],
        "category": "技术对比"
    },
    {
        "question": "RAG 系统的四个核心环节是什么？",
        "expected_answer": "RAG 系统通常包含文档加载与分块、向量化与索引、检索与生成四个核心环节。",
        "expected_sources": ["sample_document.txt"],
        "category": "系统架构"
    },
    {
        "question": "向量数据库选型时，查询延迟是考虑因素吗？",
        "expected_answer": "是的，选择向量数据库时需要考虑数据规模、查询延迟、运维成本和一致性要求等因素。",
        "expected_sources": ["sample_document.txt"],
        "category": "选型建议"
    },
    {
        "question": "大语言模型训练的三个阶段是什么？",
        "expected_answer": "大语言模型的训练通常分为三个阶段：预训练（Pre-training）、有监督微调（Supervised Fine-Tuning, SFT）和强化学习人类反馈（RLHF）阶段。",
        "expected_sources": ["sample_document.txt"],
        "category": "训练流程"
    },
    {
        "question": "什么是角色提示？",
        "expected_answer": "角色提示（Role Prompting）是一种提示工程技巧，给模型一个身份设定，帮助模型更好地完成特定任务。",
        "expected_sources": ["sample_document.txt"],
        "category": "提示工程"
    },
    {
        "question": "提示工程能显著提升什么？",
        "expected_answer": "好的提示词可以显著提升模型在特定任务上的表现。",
        "expected_sources": ["sample_document.txt"],
        "category": "优势特点"
    },
    {
        "question": "Agent 系统的任务规划能力指的是？",
        "expected_answer": "Agent 系统的关键能力包括工具调用、任务规划和记忆管理。任务规划指 Agent 可以根据用户指令规划执行步骤。",
        "expected_sources": ["sample_document.txt"],
        "category": "核心技术"
    },
    {
        "question": "RAG 为什么可以减少幻觉？",
        "expected_answer": "RAG 系统的回答基于检索到的实际内容，而不是完全依赖模型内部知识，因此可以有效减少幻觉。",
        "expected_sources": ["sample_document.txt"],
        "category": "优势特点"
    },
    {
        "question": "选择向量数据库时，运维成本是考虑因素吗？",
        "expected_answer": "是的，选择向量数据库时需要考虑数据规模、查询延迟、运维成本和一致性要求等因素。",
        "expected_sources": ["sample_document.txt"],
        "category": "选型建议"
    },
    {
        "question": "预训练阶段的目标是预测什么？",
        "expected_answer": "预训练阶段的目标是预测下一个 token，让模型在海量的无标注文本上进行自监督学习。",
        "expected_sources": ["sample_document.txt"],
        "category": "训练流程"
    },
    {
        "question": "向量数据库专门用于什么？",
        "expected_answer": "向量数据库专门用于存储和检索高维向量。",
        "expected_sources": ["sample_document.txt"],
        "category": "概念定义"
    },
    {
        "question": "Agent 系统的记忆管理能力指的是？",
        "expected_answer": "Agent 系统的关键能力包括工具调用、任务规划和记忆管理。记忆管理指 Agent 可以存储和利用历史交互信息。",
        "expected_sources": ["sample_document.txt"],
        "category": "核心技术"
    },
    {
        "question": "选择向量数据库时，一致性要求是考虑因素吗？",
        "expected_answer": "是的，选择向量数据库时需要考虑数据规模、查询延迟、运维成本和一致性要求等因素。",
        "expected_sources": ["sample_document.txt"],
        "category": "选型建议"
    }
]
