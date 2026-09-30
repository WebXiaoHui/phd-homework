"""任务一 · 字符级 tokenizer（中文文本分类够用，无需预训练词表）。

- <pad> 固定为 id 0，<unk> 固定为 id 1，其余字符按频率降序编号。
- encode() 输出定长 LongTensor（截断 + 右侧补 pad），与模型 max_len 对齐，
  这样自检 / 训练都能直接得到 (B, T) 的 id 张量。
"""
from collections import Counter

import torch


class CharTokenizer:
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, vocab: dict, max_len: int = 200):
        """vocab: {token: id}，须包含 PAD 与 UNK。"""
        self.vocab = dict(vocab)
        self.max_len = max_len
        self.pad_id = self.vocab[self.PAD]
        self.unk_id = self.vocab[self.UNK]
        self.itos = {v: k for k, v in self.vocab.items()}

    def __len__(self):
        return len(self.vocab)

    def encode(self, text: str, max_len: int | None = None) -> torch.LongTensor:
        """文本 -> 定长 id 序列（截断 + 右补 pad），形状 (max_len,)。"""
        max_len = max_len or self.max_len
        ids = [self.vocab.get(ch, self.unk_id) for ch in text[:max_len]]
        if len(ids) < max_len:
            ids += [self.pad_id] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def tokens_of(self, text: str, max_len: int | None = None):
        """返回去掉 <pad> 后的字符列表，可视化画坐标轴用。"""
        max_len = max_len or self.max_len
        return list(text[:max_len])

    @classmethod
    def build(cls, texts, max_len: int = 200,
              min_freq: int = 1, max_vocab: int | None = None) -> "CharTokenizer":
        """从文本列表统计字符频率建词表。

        min_freq：出现次数低于该值的字符统一归 <unk>，可抑制异常字符噪声。
        max_vocab：可选词表上限。
        """
        counter = Counter()
        for t in texts:
            counter.update(t)

        vocab = {cls.PAD: 0, cls.UNK: 1}
        for ch, freq in counter.most_common():
            if freq < min_freq:
                break
            if max_vocab is not None and len(vocab) >= max_vocab:
                break
            vocab[ch] = len(vocab)
        return cls(vocab, max_len)
