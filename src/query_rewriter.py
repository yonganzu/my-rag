"""
查询改写模块

独立负责将用户查询基于对话历史进行改写，解决代词指代问题。
支持两种策略：
  - keyword:  基于关键词匹配的快速规则改写（无 LLM 调用）
  - llm:      使用 LLM 进行语义理解改写（更准确，但需要 API 调用）
"""

import re
from typing import Optional

from src.config import config
from src.llm import llm_call


class QueryRewriter:
    """查询改写器：基于对话历史解决代词/指代消解"""

    def __init__(self, strategy: Optional[str] = None):
        """
        Args:
            strategy: 改写策略，可选 "keyword" / "llm"，默认从 config 读取
        """
        self.strategy = strategy or getattr(config, "query_rewrite_strategy", "keyword")

    # ── 公共接口 ────────────────────────────────────────────

    def rewrite(self, question: str, history: str) -> str:
        """根据对话历史改写用户查询"""
        if not history:
            return question

        if self.strategy == "llm":
            return self._rewrite_by_llm(question, history)
        else:
            return self._rewrite_by_keyword(question, history)

    # ── 策略一: 关键词规则改写 ─────────────────────────────

    # 中文指代词 → 从历史中提取的实体映射规则
    # 格式: (正则模式, 替换模板), 替换模板中 {entity} 会被替换为实际实体
    PRONOUN_PATTERNS = [
        (r"^(他|它|她)(.+)$", "{entity}{rest}"),          # 「他和LLM的关系」→ 「RAG和LLM的关系」
        (r"^这个(.+)$", "{entity}{rest}"),               # 「这个怎么用」→ 「RAG怎么用」
        (r"^那个(.+)$", "{entity}{rest}"),               # 「那个是什么」→ 「RAG是什么」
        (r"^(这|那)个$", "{entity}"),                    # 「这个」→ 「RAG」
    ]

    @staticmethod
    def _extract_last_entity(history: str) -> Optional[str]:
        """从对话历史最后一条助手回复中提取核心实体（首句主语）"""
        assistant_matches = re.findall(r"助手[：:]\s*(.+?)(?:\n|用户[：:]|\Z)", history, re.DOTALL)
        if not assistant_matches:
            return None

        last_reply = assistant_matches[-1].strip()
        # 取第一句话，去掉括号内的英文注释
        first_sentence = re.split(r"[。.；;！!]", last_reply)[0]
        first_sentence = re.sub(r"\（[^）]*\）|\([^)]*\)", "", first_sentence).strip()

        # 提取定义句式中的核心实体词（首个单词/缩写）
        # 「RAG（...）是一种...」→ RAG
        # 「大语言模型（LLM）是...」→ 大语言模型
        match = re.match(r"^([A-Za-z]+|[\u4e00-\u9fff]+(?:[\u4e00-\u9fff]+)*)", first_sentence)
        if match:
            entity = match.group(1)
            if len(entity) <= 20:
                return entity

        return None

    def _rewrite_by_keyword(self, question: str, history: str) -> str:
        """基于关键词规则快速改写（不调用 LLM）"""
        entity = self._extract_last_entity(history)
        if not entity:
            return question

        for pattern, template in self.PRONOUN_PATTERNS:
            match = re.match(pattern, question)
            if match:
                groups = match.groups()
                # 构建替换变量
                vars_dict = {"entity": entity}
                if "rest" in template:
                    # 找到最后一个非代词组的剩余内容
                    rest = groups[-1] if groups[-1] != entity else ""
                    vars_dict["rest"] = rest
                rewritten = template.format(**vars_dict)
                print(f"[查询改写-关键词] '{question}' → '{rewritten}'")
                return rewritten

        return question

    # ── 策略二: LLM 语义改写 ───────────────────────────────

    REWRITE_PROMPT = """你是一个查询改写助手。请根据对话历史，将用户的模糊问题改写为明确的问题，解决代词指代不明确的问题。

对话历史：
{history}

用户当前问题：{question}

改写规则：
1. 将"他"、"它"、"她"、"这个"、"那个"等指代词替换为具体的指代对象
2. 保持原问题的意思不变
3. 如果问题已经很明确，直接返回原问题
4. 只输出改写后的问题，不要输出任何解释

改写后的问题："""

    def _rewrite_by_llm(self, question: str, history: str) -> str:
        """使用 LLM 进行语义改写"""
        prompt = self.REWRITE_PROMPT.format(history=history, question=question)

        try:
            resolved = llm_call(
                model=config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                api_key=config.dashscope_api_key,
                base_url=config.llm_base_url,
            )
            resolved = resolved.strip()
            print(f"[查询改写-LLM] '{question}' → '{resolved}'")
            return resolved
        except Exception as e:
            print(f"[查询改写-LLM] 失败: {e}，回退到关键词改写")
            return self._rewrite_by_keyword(question, history)


# ── 模块级便捷函数 ────────────────────────────────────────

_default_rewriter: Optional[QueryRewriter] = None


def rewrite_query(question: str, history: str, strategy: Optional[str] = None) -> str:
    """便捷函数：改写查询"""
    global _default_rewriter
    if _default_rewriter is None or strategy is not None:
        _default_rewriter = QueryRewriter(strategy=strategy)
    return _default_rewriter.rewrite(question, history)
