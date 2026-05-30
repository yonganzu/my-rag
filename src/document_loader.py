"""
文档加载与分块模块

为什么需要分块（Chunking）？
  - LLM 有上下文窗口限制，不能把整本书塞进一个 prompt
  - 检索时，细粒度的块比整篇文档更容易匹配到用户问题
  - 块与块之间保留重叠（overlap），避免关键信息恰好被切在边界上

工程要点：
  - 用 pathlib.Path 处理路径（跨平台，比 os.path 更现代）
  - 用 try/except 包裹 IO 操作，给用户清晰的错误信息
"""

from pathlib import Path
from typing import List


def load_text(file_path: str | Path) -> str:
    """读取 txt 文件内容"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文档不存在: {path}")
    # encoding='utf-8' 显式指定，避免 Windows 默认编码问题
    return path.read_text(encoding="utf-8")


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    将文本分割成重叠的块

    参数：
      chunk_size: 每块的最大字符数
      overlap:    相邻块之间的重叠字符数

    返回：
      字符串列表，每个元素是一个文本块
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size 必须大于 overlap")

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        # 计算当前块的结束位置
        end = start + chunk_size

        # 最后一块直接取到末尾
        if end >= text_len:
            chunks.append(text[start:])
            break

        # ── 优雅切分：尽量在句号/换行处断开，而不是生硬截断 ──
        # 在当前块的最后 1/4 范围内寻找最后一个句子边界
        search_start = max(start + chunk_size * 3 // 4, start)
        # 从右向左找句号、问号、感叹号、换行
        cut = -1
        for sep in ("。", "！", "？", "\n", ".", "!", "?"):
            pos = text.rfind(sep, search_start, end)
            if pos > cut:
                cut = pos

        # 如果找到了句子边界，在边界处断开（+1 保留标点）
        if cut > search_start:
            chunks.append(text[start: cut + 1])
            start = cut + 1 - overlap
        else:
            # 找不到好边界就直接按长度切
            chunks.append(text[start:end])
            start = end - overlap

    # 防止 overlap 导致 start 回退到负数
    if start < 0:
        chunks[-1] = text  # 罕见情况：直接返回全文

    return chunks


def load_and_chunk(file_path: str | Path, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """加载文档并分块的一站式函数"""
    text = load_text(file_path)
    chunks = chunk_text(text, chunk_size, overlap)
    return chunks


def load_documents_from_folder(folder_path: str | Path, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    加载文件夹中的所有 txt 文档并分块

    参数：
      folder_path: 文件夹路径
      chunk_size: 每块的最大字符数
      overlap:    相邻块之间的重叠字符数

    返回：
      所有文档分块后的字符串列表
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"文档文件夹不存在: {folder}")

    all_chunks: List[str] = []
    txt_files = list(folder.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(f"文件夹中没有找到 .txt 文件: {folder}")

    for txt_file in txt_files:
        print(f"  加载文档: {txt_file.name}")
        chunks = load_and_chunk(txt_file, chunk_size, overlap)
        all_chunks.extend(chunks)
        print(f"    -> 分割为 {len(chunks)} 个文本块")

    return all_chunks
