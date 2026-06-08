"""
LLM（大语言模型）模块

支持的后端：
  - DashScope API：通义千问系列模型（qwen-plus, qwen-turbo 等）
  - Ollama：本地运行的开源模型（llama3, qwen, mistral 等）
  - 本地模型：HuggingFace Transformers 模型

使用方式：
  1. DashScope（默认）：设置 DASHSCOPE_API_KEY 环境变量
  2. Ollama：安装 Ollama 并运行 `ollama run qwen`，设置 LLM_TYPE=ollama
  3. 本地模型：安装 transformers，设置 LLM_TYPE=local
"""

from typing import Optional, Dict, Any

from src.config import config


# ── Ollama 客户端缓存 ───────────────────────────────────────────
_ollama_client = None


def _get_ollama_client():
    """获取 Ollama 客户端（懒加载）"""
    global _ollama_client
    if _ollama_client is None:
        try:
            import ollama
            _ollama_client = ollama
            print(f"[LLM] Ollama 客户端已加载")
        except ImportError:
            raise ImportError("请安装 ollama 库: pip install ollama")
    return _ollama_client


def _call_ollama(
    model: str,
    messages: list,
    **kwargs,
) -> str:
    """
    调用 Ollama 本地模型
    
    参数：
      model: 模型名称（如 "qwen", "llama3", "mistral"）
      messages: 消息列表，格式: [{"role": "user", "content": "..."}]
      **kwargs: 其他参数（temperature, max_tokens 等）
    
    返回：
      模型生成的文本
    """
    ollama = _get_ollama_client()
    
    print(f"[LLM] 调用 Ollama 模型: {model}")
    
    response = ollama.chat(
        model=model,
        messages=messages,
        options={
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 4096),
        },
    )
    
    return response["message"]["content"]


def _call_dashscope(
    model: str,
    messages: list,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
) -> str:
    """
    调用 DashScope API
    
    参数：
      model: 模型名称（如 "qwen-plus", "qwen-turbo"）
      messages: 消息列表，格式: [{"role": "user", "content": "..."}]
      api_key: DashScope API Key
      base_url: API 基础 URL
      **kwargs: 其他参数
    
    返回：
      模型生成的文本
    """
    from dashscope import Generation
    
    print(f"[LLM] 调用 DashScope API: {model}")
    
    resp = Generation.call(
        model=model,
        messages=messages,
        api_key=api_key,
        base_url=base_url,
        result_format="message",
        **kwargs,
    )
    
    if resp.status_code != 200:
        raise RuntimeError(f"LLM API 调用失败: [{resp.status_code}] {resp.message}")
    
    return resp.output.choices[0].message.content


def _call_local_transformers(
    model: str,
    messages: list,
    **kwargs,
) -> str:
    """
    调用本地 Transformers 模型
    
    参数：
      model: 模型路径或 HuggingFace 模型名称
      messages: 消息列表
      **kwargs: 其他参数
    
    返回：
      模型生成的文本
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    except ImportError:
        raise ImportError("请安装 transformers: pip install transformers torch")
    
    print(f"[LLM] 加载本地 Transformers 模型: {model}")
    
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model)
    
    # 构建对话格式
    conversation = ""
    for msg in messages:
        if msg["role"] == "user":
            conversation += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            conversation += f"Assistant: {msg['content']}\n"
    
    conversation += "Assistant:"
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    response = pipe(
        conversation,
        max_new_tokens=kwargs.get("max_tokens", 512),
        temperature=kwargs.get("temperature", 0.7),
        do_sample=True,
    )
    
    return response[0]["generated_text"].replace(conversation, "").strip()


def llm_call(
    model: str,
    messages: list,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    llm_type: Optional[str] = None,
    **kwargs,
) -> str:
    """
    统一的 LLM 调用接口
    
    参数：
      model: 模型名称
      messages: 消息列表，格式: [{"role": "user", "content": "..."}]
      api_key: API Key（DashScope 使用）
      base_url: API 基础 URL
      llm_type: LLM 类型 ("dashscope", "ollama", "local")，默认为配置值
      **kwargs: 其他参数（temperature, max_tokens 等）
    
    返回：
      模型生成的文本
    """
    if llm_type is None:
        llm_type = config.llm_type
    
    if llm_type == "ollama":
        return _call_ollama(model, messages, **kwargs)
    elif llm_type == "local":
        return _call_local_transformers(model, messages, **kwargs)
    else:
        return _call_dashscope(model, messages, api_key=api_key, base_url=base_url, **kwargs)
