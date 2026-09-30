"""任务二 · RoPE 旋转位置编码（Rotary Position Embedding）。

要点：
- 只作用在 Q / K 上，**不作用在 V**；
- 维度采用"前后折半"配对（dim i 与 dim i + head_dim//2 成一组旋转），
  与 LLaMA / 常见实现一致，只需保证 Q 和 K 用同一约定；
- 频率基频 base=10000；
- 每个位置按**绝对位置**算 (cos, sin)，因此支持任意起始 offset ——
  增量解码（KV cache）时新 token 的位置 = 历史长度 + i，必须按此计算，
  否则角度错、和全量前向对不上。
"""
import torch


def rotary_freqs(positions: torch.Tensor, head_dim: int, base: float = 10000.0):
    """按绝对位置算 cos / sin。

    positions: (T,) 的绝对位置（长整型）；
    返回 (cos, sin)，形状各为 (T, head_dim//2)。
    """
    # 每个"半对"一个频率：dim0,2,4... 共 head_dim//2 个
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2,
                                            device=positions.device,
                                            dtype=torch.float32) / head_dim))
    angles = positions.float().unsqueeze(1) * inv_freq.unsqueeze(0)  # (T, half)
    return torch.cos(angles), torch.sin(angles)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor,
                         sin: torch.Tensor) -> torch.Tensor:
    """把旋转作用到 x（shape (B, H, T, D)，D 须为偶数）。

    cos / sin: (T, D//2)，自动广播到 batch 与 head 维。
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    # 补两个前导维，使 (T, D//2) 广播到 (B, H, T, D//2)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin,
                      x1 * sin + x2 * cos], dim=-1)
