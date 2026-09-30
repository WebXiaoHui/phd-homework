"""task-4 · 稠密检索 Retriever（BGE embedding + FAISS）。

对外契约（README / eval/run.py）：
    class Retriever:
        retrieve(query: str, k: int) -> List[dict]   # 每项含 text / score / source

Retriever() 无参即可用：构造时加载 src.paths.INDEX_DIR 的索引 + bge 模型。
索引不存在会直接报错（提示先跑 build_index.py），而不是静默返回空结果——
自检 nndl_gold_recall_at_10 才能反映真实状态。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from src.paths import INDEX_DIR

try:
    import faiss
except ImportError:
    faiss = None


class Retriever:
    def __init__(self, index_dir: Union[str, Path] = INDEX_DIR):
        self.index_dir = Path(index_dir)
        _ensure_index(self.index_dir)
        if faiss is None:
            raise ImportError("需要 faiss-cpu：pip install -r requirements.txt")

        with open(self.index_dir / "index.faiss", "rb") as f:
            data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        self._index = faiss.deserialize_index(arr)
        self.chunks = json.loads(
            (self.index_dir / "chunks.json").read_text(encoding="utf-8"))
        meta = json.loads(
            (self.index_dir / "meta.json").read_text(encoding="utf-8"))
        self.meta = meta
        if len(self.chunks) != self._index.ntotal:
            print(f"[retriever] 警告：chunks={len(self.chunks)} 与 "
                  f"faiss ntotal={self._index.ntotal} 不一致，请重建索引")

    # ---- 检索 ----

    def retrieve(self, query: str, k: int = 10) -> List[dict]:
        """稠密检索 top-k。k 生效（不做成写死值）。"""
        from src.embed import embed_query

        if k <= 0 or not self.chunks:
            return []
        q = embed_query(query)                      # bge 自动加检索前缀+归一化
        kk = min(k, len(self.chunks))
        scores, idxs = self._index.search(
            np.ascontiguousarray(q.reshape(1, -1), dtype=np.float32), kk)
        out = []
        for score, j in zip(scores[0].tolist(), idxs[0].tolist()):
            if j < 0 or j >= len(self.chunks):
                continue
            c = self.chunks[j]
            out.append({"text": c.get("text", ""),
                        "score": float(score),
                        "source": c.get("source", ""),
                        "page": c.get("page")})
        return out

    # ---- 便捷 ----

    def __len__(self):
        return len(self.chunks)


def _ensure_index(index_dir: Path):
    need = [index_dir / "index.faiss", index_dir / "chunks.json",
            index_dir / "meta.json"]
    missing = [p.name for p in need if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"索引不完整（缺 {missing}）：{index_dir}\n"
            "请先运行：cd task-4-rag && python build_index.py\n"
            "（它会从 data/kb.pdf 抽文本 -> BGE embedding -> FAISS）")
