"""task-4 · BGE embedding 封装（含官方检索前缀 / 归一化）。

BGE 用法要点（README 常见坑）：
- 查询要加检索前缀：`为这个句子生成表示以用于检索相关文章：`，文档侧不加；
- 用余弦距离时先做 L2 归一化（本封装 normalize_embeddings=True），这样
  FAISS 内积 == 余弦相似度。
模型只加载一次（进程级缓存），检索 30 条 gold QA 不会反复 reload。
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.paths import EMB_MODEL_DIR

QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

_model_cache = None


def _load_model():
    global _model_cache
    if _model_cache is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "需要 sentence-transformers：pip install -r requirements.txt"
            ) from e
        if not EMB_MODEL_DIR.exists():
            raise FileNotFoundError(
                f"未找到 embedding 模型：{EMB_MODEL_DIR}\n"
                "先运行 python data/download.py（会拉 BAAI/bge-small-zh-v1.5）")
        _model_cache = SentenceTransformer(str(EMB_MODEL_DIR))
    return _model_cache


def encode_texts(texts: List[str], is_query: bool = False,
                 batch_size: int = 32, show_progress: bool = False
                 ) -> np.ndarray:
    """把文本批量编码成 (n, dim) float32 向量（已归一化）。

    is_query=True 时自动加 bge 检索前缀。
    """
    model = _load_model()
    inputs = [QUERY_PREFIX + t for t in texts] if is_query else list(texts)
    if not inputs:
        return np.zeros((0, model.get_sentence_embedding_dimension()),
                        dtype=np.float32)
    vecs = model.encode(inputs, batch_size=batch_size, show_progress_bar=show_progress,
                        normalize_embeddings=True, convert_to_numpy=True)
    return np.asarray(vecs, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    return encode_texts([text], is_query=True)[0]


def embed_documents(texts: List[str], batch_size: int = 32) -> np.ndarray:
    return encode_texts(texts, is_query=False, batch_size=batch_size)
