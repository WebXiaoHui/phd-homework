"""task-4 · 索引：PDF -> chunk -> BGE embedding -> FAISS（内积，等价余弦）。

**索引必须来自 data/kb.pdf 抽取的文本**（不允许索引 LaTeX 源，否则违背任务）。
产物写进 data/index/：
    chunks.json      chunk 记录列表 [{text, source, page}]
    embeddings.npy   已归一化向量 (N, dim) float32（便于复现/调试）
    index.faiss      FAISS 内积索引（向量已归一化 => 内积即余弦）
    meta.json        chunk_size / overlap / dim / 模型名 等
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Union

import numpy as np

from src.chunker import chunk_pdf
from src.embed import embed_documents
from src.paths import INDEX_DIR, KB_PDF

try:
    import faiss
except ImportError:                       # faiss 可选缺失时保留清晰报错
    faiss = None


def build_index(pdf_path: Union[str, Path], chunk_size: int = 512,
                overlap: int = 128, index_dir: Union[str, Path] = INDEX_DIR,
                force: bool = False, batch_size: int = 32) -> dict:
    """跑完整条索引管线，返回 {index_path, n_chunks, dim, elapsed_s}。"""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"缺少知识库 PDF：{pdf_path}；先运行 python data/download.py")
    if faiss is None:
        raise ImportError("需要 faiss-cpu：pip install -r requirements.txt")
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    if (index_dir / "index.faiss").exists() and not force:
        print(f"[indexer] 已存在索引 {index_dir}（用 --force 重建）")
        return _summarize(index_dir)

    t0 = time.time()
    print(f"[indexer] 抽取 PDF 文本：{pdf_path.name}")
    chunks: List[dict] = chunk_pdf(pdf_path, chunk_size=chunk_size,
                                   overlap=overlap)
    if len(chunks) < 10:
        raise RuntimeError(f"只抽到 {len(chunks)} 个 chunk，PDF 文本抽取可能失败")

    texts = [c["text"] for c in chunks]
    print(f"[indexer] {len(chunks)} chunks -> embedding "
          f"(chunk_size={chunk_size}, overlap={overlap})")
    vecs = embed_documents(texts, batch_size=batch_size)

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)          # 向量已归一化，内积=余弦
    index.add(np.ascontiguousarray(vecs, dtype=np.float32))

    (index_dir / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    np.save(index_dir / "embeddings.npy", vecs)
    # 序列化到内存，再用 Python 的 open 写文件（绕过 FAISS 对中文路径的限制）
    index_bytes = faiss.serialize_index(index)
    with open(index_dir / "index.faiss", "wb") as f:
        f.write(index_bytes.tobytes())
    (index_dir / "meta.json").write_text(
        json.dumps({"chunk_size": chunk_size, "overlap": overlap, "dim": dim,
                    "n_chunks": len(chunks),
                    "embed_model": str(EMB_MODEL_NAME())}, ensure_ascii=False,
                   indent=2), encoding="utf-8")
    el = time.time() - t0
    print(f"[indexer] 完成：{len(chunks)} chunks -> {index_dir}（{el:.1f}s）")
    return {"index_path": str(index_dir), "n_chunks": len(chunks),
            "dim": dim, "elapsed_s": round(el, 1)}


def EMB_MODEL_NAME():
    from src.paths import EMB_MODEL_DIR
    return EMB_MODEL_DIR.name


def _summarize(index_dir: Path) -> dict:
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    return {"index_path": str(index_dir),
            "n_chunks": meta.get("n_chunks"),
            "dim": meta.get("dim"),
            "chunk_size": meta.get("chunk_size")}


def main():
    ap = argparse.ArgumentParser(description="建立 data/kb.pdf 的 RAG 索引")
    ap.add_argument("--pdf", default=str(KB_PDF))
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--index-dir", default=str(INDEX_DIR))
    ap.add_argument("--force", action="store_true",
                    help="索引已存在也重建（改 chunk 参数时用）")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    build_index(args.pdf, chunk_size=args.chunk_size, overlap=args.overlap,
                index_dir=args.index_dir, force=args.force,
                batch_size=args.batch_size)


if __name__ == "__main__":
    main()
