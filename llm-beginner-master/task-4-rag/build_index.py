"""task-4 · 一键建索引：data/kb.pdf -> data/index/（chunks + embedding + FAISS）。

用法：
    python build_index.py                      # 默认 chunk_size=512, overlap=128
    python build_index.py --chunk-size 256 --overlap 64 --force   # 消融/重建
"""
import argparse

from src.indexer import build_index
from src.paths import INDEX_DIR, KB_PDF


def main():
    ap = argparse.ArgumentParser(description="从 data/kb.pdf 建立 RAG 索引")
    ap.add_argument("--pdf", default=str(KB_PDF))
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--index-dir", default=str(INDEX_DIR))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    build_index(args.pdf, chunk_size=args.chunk_size, overlap=args.overlap,
                index_dir=args.index_dir, batch_size=args.batch_size,
                force=args.force)


if __name__ == "__main__":
    main()
