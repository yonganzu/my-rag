"""
BM25 关键词检索器（独立模块）

提供纯关键词检索能力，可与向量检索配合使用。

BM25 公式：
  score(D, Q) = Σ IDF(qi) × (TF(qi,D) × (k1+1)) / (TF(qi,D) + k1×(1-b+b×|D|/avgdl))

参数说明：
  - k1: 词频饱和度（防止词频过高主导评分），默认 1.5
  - b: 文档长度归一化（长文档不天然占优），默认 0.75
"""

import math
import re
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional


def _tokenize(text: str) -> List[str]:
    """简单分词：中英文混合处理"""
    try:
        import jieba
    except ImportError:
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+', text.lower())
        return [w for w in words if len(w) > 1]

    chinese_parts = re.findall(r'[\u4e00-\u9fff]+', text)
    for part in chinese_parts:
        words = list(jieba.cut(part))
        text = text.replace(part, ' '.join(words))
    tokens = re.findall(r'[a-zA-Z0-9]+|[\u4e00-\u9fff]+', text.lower())
    return [t for t in tokens if len(t) > 1]


class BM25Retriever:
    """
    BM25 关键词检索器
    
    独立的关键词检索模块，可与向量检索配合使用。
    
    使用示例：
        retriever = BM25Retriever()
        retriever.build(["文档1内容", "文档2内容"])
        results = retriever.search("查询词", top_k=3)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[str] = []  # 原始文档文本
        self.tokens_list: List[List[str]] = []  # 分词结果
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0
        self.idf: Dict[str, float] = {}
        self._doc_count: int = 0
        self._term_doc_freq: Dict[str, int] = {}
        self.sources: List[str] = []  # 文档来源

    def build(self, documents: List[str], sources: Optional[List[str]] = None) -> None:
        """构建 BM25 索引"""
        self.documents = documents
        self.tokens_list = []
        self.doc_lengths = []
        self._term_doc_freq = {}
        # 确保 sources 长度与 documents 匹配，不足部分用默认值填充
        if sources and len(sources) >= len(documents):
            self.sources = sources[:len(documents)]
        else:
            self.sources = list(sources) if sources else []
            while len(self.sources) < len(documents):
                self.sources.append("未知来源")

        for doc in documents:
            tokens = _tokenize(doc)
            self.tokens_list.append(tokens)
            self.doc_lengths.append(len(tokens))
            unique_terms = set(tokens)
            for term in unique_terms:
                self._term_doc_freq[term] = self._term_doc_freq.get(term, 0) + 1

        self._doc_count = len(self.documents)
        if self._doc_count == 0:
            return

        self.avgdl = sum(self.doc_lengths) / self._doc_count

        for term, df in self._term_doc_freq.items():
            self.idf[term] = math.log(
                (self._doc_count - df + 0.5) / (df + 0.5) + 1.0
            )

    def add_documents(self, documents: List[str], sources: Optional[List[str]] = None) -> None:
        """增量添加文档（自动去重，避免重复添加导致词频错误）"""
        sources = sources or []
        added_count = 0

        for i, doc in enumerate(documents):
            # 去重检查：如果文档内容已存在则跳过
            if doc in self.documents:
                continue

            tokens = _tokenize(doc)
            self.tokens_list.append(tokens)
            self.doc_lengths.append(len(tokens))
            self.documents.append(doc)

            # 同步添加来源信息
            if i < len(sources):
                self.sources.append(sources[i])
            else:
                self.sources.append("未知来源")

            # 更新词项文档频率
            unique_terms = set(tokens)
            for term in unique_terms:
                self._term_doc_freq[term] = self._term_doc_freq.get(term, 0) + 1

            added_count += 1

        if added_count == 0:
            print("[BM25] 没有新文档需要添加（全部已存在）")
            return

        self._doc_count = len(self.documents)
        self.avgdl = sum(self.doc_lengths) / self._doc_count

        # 重新计算所有 term 的 IDF
        for term, df in self._term_doc_freq.items():
            self.idf[term] = math.log(
                (self._doc_count - df + 0.5) / (df + 0.5) + 1.0
            )

        print(f"[BM25] 已添加 {added_count} 个新文档，共 {self._doc_count} 个文档")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[int, float, str, str]]:
        """
        BM25 检索
        
        参数：
          query: 查询文本
          top_k: 返回前 k 个结果
        
        返回：
          [(doc_index, bm25_score, document_text, source), ...]
        """
        if self._doc_count == 0:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = [0.0] * self._doc_count
        for token in set(query_tokens):
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for i, doc_tokens in enumerate(self.tokens_list):
                tf = doc_tokens.count(token)
                if tf == 0:
                    continue
                doc_len = self.doc_lengths[i]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                scores[i] += idf_val * numerator / denominator

        indexed_scores = [(i, scores[i]) for i in range(self._doc_count)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in indexed_scores[:top_k]:
            if score > 0:
                source = self.sources[idx] if idx < len(self.sources) else "未知来源"
                results.append((idx, score, self.documents[idx], source))

        return results

    def save(self, dir_path: str) -> None:
        """持久化索引到磁盘"""
        save_dir = Path(dir_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "bm25_index.json"
        data = {
            "k1": self.k1,
            "b": self.b,
            "documents": self.documents,
            "tokens_list": self.tokens_list,
            "idf": self.idf,
            "doc_lengths": self.doc_lengths,
            "avgdl": self.avgdl,
            "term_doc_freq": self._term_doc_freq,
            "doc_count": self._doc_count,
            "sources": self.sources,
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[BM25] 索引已保存到: {save_path}")

    def load(self, dir_path: str) -> bool:
        """从磁盘加载索引"""
        load_dir = Path(dir_path)
        path = load_dir / "bm25_index.json"
        if not path.exists():
            print(f"[BM25] 索引文件不存在: {path}")
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.k1 = data["k1"]
            self.b = data["b"]
            
            # 兼容旧格式：旧版本中 documents 存储的是分词结果，没有 tokens_list
            if "tokens_list" in data:
                self.tokens_list = data["tokens_list"]
                self.documents = data["documents"]
            else:
                # 旧格式：documents 是分词后的列表，需要转换
                print(f"[BM25] 检测到旧格式索引，正在转换...")
                self.tokens_list = data["documents"]
                # 从分词重建原始文档文本
                self.documents = [" ".join(tokens) for tokens in self.tokens_list]
            
            self.idf = data["idf"]
            self.doc_lengths = data["doc_lengths"]
            self.avgdl = data["avgdl"]
            self._term_doc_freq = data["term_doc_freq"]
            self._doc_count = data["doc_count"]
            self.sources = data.get("sources", [])
            print(f"[BM25] 索引加载成功，共 {self._doc_count} 个文档")
            return True
        except Exception as e:
            print(f"[BM25] 索引加载失败: {e}")
            return False

    def __len__(self) -> int:
        """返回文档数量"""
        return self._doc_count

    def is_empty(self) -> bool:
        """判断是否为空"""
        return self._doc_count == 0
