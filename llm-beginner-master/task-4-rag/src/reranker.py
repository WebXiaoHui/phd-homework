"""task-4 · BGE 交叉编码器精排（bge-reranker-base）。

reranker 的输入是 [query, doc] 文本对（不是 embedding），对召回的多段再做
一次细粒度打分后取 top_k。通常「召回 20 -> rerank 取 top 5」。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from src.paths import RERANK_MODEL_DIR


class Reranker:
    def __init__(self, model_dir: Union[str, Path] = RERANK_MODEL_DIR,
                 device: str = "auto"):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"未找到 reranker 模型：{self.model_dir}\n"
                "先运行 python data/download.py；或给 rag 传 reranker=False 跳过")
        self.device = _resolve_device(device)
        self._model = None
        self._tok = None

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(str(self.model_dir))
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir))
        self._model.to(self.device)
        self._model.eval()

    def rerank(self, query: str, docs: List[dict],
               top_k: Optional[int] = None) -> List[dict]:
        """对 docs（每项含 text）按 query 精排，返回新列表，倒序、附 rerank_score。"""
        import torch

        if not docs:
            return []
        self._ensure()
        pairs = [[query, d.get("text", "")] for d in docs]
        inputs = self._tok(
            pairs, padding=True, truncation=True, max_length=512,
            return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits.flatten().tolist()
        ranked = sorted(zip(docs, logits), key=lambda x: x[1], reverse=True)
        out = []
        for d, s in ranked:
            item = dict(d)
            item["rerank_score"] = float(s)
            out.append(item)
        if top_k is not None:
            out = out[:top_k]
        return out


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
