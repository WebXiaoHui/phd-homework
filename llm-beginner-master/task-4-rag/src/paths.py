"""task-4 · 统一路径约定（PDF / 索引 / 模型都在固定位置）。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KB_PDF = ROOT / "data" / "kb.pdf"                 # 知识库 PDF（唯一允许的知识来源）
GOLD_QA = ROOT / "data" / "gold_qa.jsonl"         # 评测题目（30 条 NNDL gold QA）
INDEX_DIR = ROOT / "data" / "index"               # FAISS 索引产物
MODELS_DIR = ROOT / "models"
EMB_MODEL_DIR = MODELS_DIR / "bge-small-zh-v1.5"  # 向量模型
RERANK_MODEL_DIR = MODELS_DIR / "bge-reranker-base"  # 精排模型

# 默认 RAG 参数（可用环境变量覆盖）
DEFAULT_CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "512"))
DEFAULT_OVERLAP = int(os.environ.get("RAG_OVERLAP", "128"))
DEFAULT_TOP_K_FINAL = 5
DEFAULT_TOP_K_RECALL = 20          # 召回数 >> 最终返回数，rerank 才有得选
