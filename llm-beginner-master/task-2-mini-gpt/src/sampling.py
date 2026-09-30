"""任务二 · 采样策略：greedy / top-k / top-p / temperature。

约定：
- temperature <= 0 视作 greedy（取 argmax），避免除零；
- top-k / top-p 都基于"处理后"的 logits / probs，truncate 后要**重新归一化**；
- 先 temp -> top-k -> softmax -> top-p -> multinomial。
"""
import torch
import torch.nn.functional as F


def sample_token(logits, temperature=1.0, top_k=None, top_p=None) -> int:
    """给定一个 token 的 logits（一维或带前导批维），返回采样出的 token id。

    top_k: 保留概率最高的 top_k 个候选（None / <=0 表示不启用）。
    top_p: 核采样阈值，0<p<1（None / <=0 表示不启用）。
    """
    if logits.dim() > 1:
        logits = logits.flatten(0, -2)          # 展平批/位置维，通常只有一个

    logits = logits.float()

    # temperature <= 0 => greedy
    if temperature is not None and temperature <= 0:
        return int(logits.argmax(dim=-1).item())

    if temperature is not None:
        logits = logits / temperature

    # top-k：把阈值以下的 logits 置 -inf（相对较小的 top_k 值以下的都干掉）
    if top_k is not None and top_k > 0 and logits.numel() > top_k:
        kth = torch.topk(logits, top_k).values[..., -1]     # 第 top_k 大的值
        logits = torch.where(logits < kth, float("-inf"), logits)

    probs = F.softmax(logits, dim=-1)

    # top-p：按概率降序累加，一旦超过阈值就把后面（超额）的候选置 0
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        # 当前项之前的累计已经超过阈值 => 移除当前项
        remove = (cumsum - sorted_probs) > top_p
        probs = probs.clone()
        probs[sorted_idx[remove]] = 0.0
        denom = probs.sum()
        if denom > 0:
            probs = probs / denom
        else:
            probs = torch.softmax(logits, dim=-1)

    return int(torch.multinomial(probs, num_samples=1).item())
