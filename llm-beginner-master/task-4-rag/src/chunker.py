"""task-4 · 文本切分（chunking）。

对外契约（README / eval/run.py）：
    chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]

约定：
- chunk_size / overlap **以字符计**，不是词元数；自检按「平均字符长度在
  (chunk_size*0.5, chunk_size*1.2)」核验；
- 尽量在句末标点（。！？!?；;）或换行后断开，避免在句子中间硬切；
- 相邻 chunk 用 overlap 个字符重叠，避免 gold anchor 恰好落在切割缝上。

另提供 PDF 侧工具：extract_pages（逐页抽文本，供 indexer 使用，加 source 页号）。
抽取文本常有换行碎片/表格乱序，本模块不做过多语义处理（中文句内换行会在
clean 时拼回），语义/表格优化留给你在报告里说明取舍。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Union

# 适合断句的边界：这些字符应当留在 chunk 尾部
_SENT_END = "。！？!?；;…"
# 额外允许断开的换行/分节
_BREAK = _SENT_END + "\n"

RE_WS = re.compile(r"[ \t\r\f\v　]+")
RE_MULTI_NL = re.compile(r"\n{2,}")


def clean_page_text(text: str) -> str:
    """把一页 PDF 抽取文本整理成适合切分的形式。

    - 全角空格/连续空格压成单个空格，连续空行压成一段的分隔；
    - 中文句子中间的换行（行尾是文字/标点、行首是汉字）直接去掉——PDF 正文
      在页面里按行断行，这些换行不是真实段落边界。
    """
    if not text:
        return ""
    text = text.replace("　", " ")
    text = RE_MULTI_NL.sub("\n\n", text)
    text = RE_WS.sub(" ", text)
    # 去掉「汉字/中文标点 结尾 + 换行 + 汉字开头」这类伪换行
    text = re.sub(
        r"(?<=[一-鿿，。！？；：、）])"
        r"\n(?=[一-鿿（“])",
        "", text)
    return text.strip()


def _last_break_index(seg: str, min_cut: int) -> Optional[int]:
    """seg 里最靠后的断句点下标；太靠近开头则返回 None（避免切出过短 chunk）。"""
    for i in range(len(seg) - 1, -1, -1):
        if seg[i] in _BREAK:
            if i >= min_cut:
                return i
            return None
    return None


def chunk_text(text: str, chunk_size: int = 256, overlap: int = 32) -> List[str]:
    """把长文本切成有重叠的 chunk 列表（字符计）。

    基本策略：滑动窗口扫过去，每个窗口若在后半段找到断句标点，就停在该标点
    后（尽量不割句子）；找不到就在 chunk_size 处硬切。overlap 让切缝两侧都
    保留一段上下文，防止锚点/语义片段被切在边界上。
    """
    if overlap < 0 or chunk_size < 1:
        raise ValueError("chunk_size >= 1 且 overlap >= 0")
    text = text.strip()
    if not text:
        return []
    n = len(text)
    if n <= chunk_size:
        return [text]

    min_cut = max(1, chunk_size // 2)
    chunks: List[str] = []
    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        seg = text[start:end]
        cut = _last_break_index(seg, min_cut)
        if cut is not None:
            end = start + cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        nxt = end - overlap
        if nxt <= start:               # overlap 过大时至少前进 1 个字符
            nxt = start + 1
        start = nxt
    return chunks


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------

def extract_pages(pdf_path: Union[str, Path]) -> List[str]:
    """用 pypdf 逐页抽文本，返回 [page_1_text, page_2_text, ...]（未 clean）。"""
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError("需要 pypdf：pip install -r requirements.txt") from e
    reader = PdfReader(str(pdf_path))
    return [(page.extract_text() or "") for page in reader.pages]


def chunk_pdf(pdf_path: Union[str, Path], chunk_size: int = 512,
              overlap: int = 128) -> List[dict]:
    """整本 PDF -> chunk 记录列表（每页独立切，便于 source 定位到页）。

    返回每个元素：{"text": str, "source": "p.N", "page": N}。
    """
    pages = extract_pages(pdf_path)
    out: List[dict] = []
    for i, raw in enumerate(pages, 1):
        text = clean_page_text(raw)
        for c in chunk_text(text, chunk_size=chunk_size, overlap=overlap):
            out.append({"text": c, "source": f"p.{i}", "page": i})
    return out
