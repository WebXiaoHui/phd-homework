"""task-4-rag：端到端中文 RAG（PDF chunking -> BGE + FAISS -> rerank -> Qwen 生成）。

组成：
- chunker.py   PDF 抽文本 + 字符级 overlap 切分
- indexer.py   建 FAISS 索引（向量已归一化，内积=余弦）
- retriever.py Retriever.retrieve(query, k)
- reranker.py  bge-reranker-base 精排
- generator.py OpenAI 兼容本地 Qwen 服务
- rag.py       answer(query) 端到端
"""
