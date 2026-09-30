"""任务二 · decoder-only mini-GPT（从零实现，集成 RoPE + KV cache）。

对外接口（与 eval/run.py 对齐）：
    class MiniGPT(nn.Module)
        forward(ids, kv_cache=None, return_cache=False)
        generate(prompt_ids, max_new_tokens, top_k, top_p, temperature)
        属性 block_size（自检按它切窗算困惑度）
    load_for_eval(ckpt_path) -> (model, tokenizer)

KV cache 语义：cache 是一份长度为 n_layer 的列表，第 l 项是第 l 层的
(k, v)（形状 (B, n_head, T_prev, d_head)），逐 token 增量时把它传回、
并取出新的 cache 继续喂，等价于全量前向（自检 kv_cache_equivalence 即验这个）。
"""
from pathlib import Path

import torch
import torch.nn as nn

from src.attention import CausalSelfAttention
from src.sampling import sample_token
from src.tokenizer import BPETokenizer


class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Pre-LN decoder block：注意力（带 cache）后接 FFN，各自 LN + residual。"""

    def __init__(self, n_embd, n_head, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout=dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout=dropout)

    def forward(self, x, cache=None):
        h = self.ln1(x)
        h, kv = self.attn(h, cache=cache)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x, kv


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=256, n_head=4, n_layer=4,
                 block_size=128, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size          # 自检靠它切窗，务必暴露

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, dropout=dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # weight tying：输入嵌入与输出头共享权重（语言建模常用正则化）
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # GPT-2 风格：残差投影的输出缩放 1/sqrt(2*n_layer)，长堆叠更稳
        for block in self.blocks:
            nn.init.normal_(block.attn.Wo.weight,
                            std=0.02 / (2 * n_layer) ** 0.5)
            nn.init.normal_(block.mlp.net[2].weight,   # FFN 的下投影层
                            std=0.02 / (2 * n_layer) ** 0.5)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids, kv_cache=None, return_cache=False):
        """ids: (B, T)。返回 logits (B, T, vocab)；
        若 return_cache=True 则返回 (logits, new_cache)。
        """
        B, T = ids.shape
        if kv_cache is not None and len(kv_cache) != self.n_layer:
            raise ValueError(f"kv_cache 长度应为 {self.n_layer}，实际 {len(kv_cache)}")

        x = self.wte(ids)                              # (B, T, n_embd)
        cache_entries = kv_cache if kv_cache is not None else [None] * self.n_layer

        new_cache = []
        for l, block in enumerate(self.blocks):
            x, kv = block(x, cache=cache_entries[l])
            new_cache.append(kv)

        logits = self.lm_head(self.ln_f(x))            # (B, T, vocab)
        return (logits, new_cache) if return_cache else logits

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=64, top_k=None,
                 top_p=None, temperature=1.0):
        """自回归生成（带 KV cache）。

        返回新生成的 token id 列表（不含 prompt）。调用前建议 model.eval()。
        top_k / top_p / temperature 语义见 src/sampling.py。
        """
        device = next(self.parameters()).device
        # 用整个 prompt 前向一次预热 cache
        prompt = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
        logits, cache = self.forward(prompt, return_cache=True)

        window_ids = list(prompt_ids)
        generated = []
        for _ in range(max_new_tokens):
            # 每步都只采样最后位置的分布
            nxt = sample_token(logits[0, -1], temperature=temperature,
                               top_k=top_k, top_p=top_p)
            generated.append(nxt)
            window_ids.append(nxt)

            if len(window_ids) >= self.block_size:
                # 超过训练上下文长度：从最近 block_size 个 token 重新预热 cache
                window = window_ids[-self.block_size:]
                x = torch.tensor([window], dtype=torch.long, device=device)
                logits, cache = self.forward(x, return_cache=True)
            else:
                x = torch.tensor([[nxt]], dtype=torch.long, device=device)
                logits, cache = self.forward(x, kv_cache=cache, return_cache=True)
        return generated


def load_for_eval(ckpt_path: str):
    """从 ckpt/best.pt 恢复模型与 tokenizer。

    best.pt 内容：{"model": state_dict, "config": {...}}；
    tokenizer 从同目录 ckpt/tokenizer.json 加载。
    """
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    tokenizer = BPETokenizer.from_pretrained(
        str(ckpt_path.parent / "tokenizer.json"))

    model = MiniGPT(vocab_size=cfg["vocab_size"], n_embd=cfg["n_embd"],
                    n_head=cfg["n_head"], n_layer=cfg["n_layer"],
                    block_size=cfg["block_size"], dropout=cfg["dropout"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer
