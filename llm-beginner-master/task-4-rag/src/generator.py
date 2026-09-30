"""task-4 · 生成器：调本地 OpenAI 兼容的 Qwen2.5-7B-Instruct 服务。

Ollama / vLLM / llama.cpp 都提供 /v1/chat/completions，这里直接用 requests
调，避免引入 openai 依赖。默认指向本地 Ollama（qwen2.5:7b-instruct），
可用环境变量覆盖：
    RAG_LLM_BASE_URL   e.g. http://localhost:11434/v1
    RAG_LLM_MODEL      e.g. qwen2.5:7b-instruct  /  Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import json
import os
from typing import List

import requests

DEFAULT_BASE_URL = os.environ.get(
    "RAG_LLM_BASE_URL", "http://localhost:11434/v1")
DEFAULT_MODEL = os.environ.get("RAG_LLM_MODEL", "qwen2.5:7b-instruct")


class Generator:
    def __init__(self, base_url: str = DEFAULT_BASE_URL,
                 model: str = DEFAULT_MODEL, timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, messages: List[dict], temperature: float = 0.2,
             max_tokens: int = 512) -> str:
        """messages: [{role, content}, ...]，返回 assistant 文本。

        服务不可达/返回异常会抛 RuntimeError（提示先起本地模型），这样
        rag_end_to_end 自检若失败能看到明确原因，而不是拿到空答案。
        """
        url = f"{self.base_url}/chat/completions"
        payload = {"model": self.model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens,
                   "stream": False}
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
        except requests.RequestException as e:
            raise RuntimeError(
                f"连不上本地生成服务 {self.base_url}：{e}\n"
                "请先启动模型服务，例如：ollama serve && ollama pull "
                "qwen2.5:7b-instruct") from e
        if resp.status_code != 200:
            raise RuntimeError(
                f"生成服务返回 {resp.status_code}：{resp.text[:300]}\n"
                f"若提示 model 不存在，改环境变量 RAG_LLM_MODEL（当前 "
                f"{self.model}）")
        try:
            return resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise RuntimeError(f"解析生成服务响应失败：{resp.text[:300]}") from e
