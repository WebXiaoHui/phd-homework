"""任务二 · 生成示例：同一 prompt 对比 greedy / top-k / top-p / temperature。

用法（先训练出 ckpt/best.pt 与 tokenizer.json）：
    python demo_generate.py --prompt "床前明月光"
    python demo_generate.py --prompt "春眠不觉晓" --max-new-tokens 60

可选 --check-cache：先做一次「KV cache 开/关 logits 是否一致」的自检
（等价于 eval/run.py 的 kv_cache_equivalence），帮助在生成前尽早发现问题。
"""
import argparse
import time

import torch

from src.model import load_for_eval

PRESETS = [
    ("greedy (t=0)", dict(temperature=0.0)),
    ("top-k=20  t=0.8", dict(temperature=0.8, top_k=20)),
    ("top-p=0.9 t=0.8", dict(temperature=0.8, top_p=0.9)),
    ("t=1.2 全采样", dict(temperature=1.2)),
]


@torch.no_grad()
def check_cache_equivalence(model, tok, text="从前有座山"):
    ids = torch.tensor([tok.encode(text)], dtype=torch.long)
    full = model(ids)
    cache = None
    inc = []
    for i in range(ids.size(1)):
        out, cache = model(ids[:, i:i + 1], kv_cache=cache, return_cache=True)
        inc.append(out)
    inc = torch.cat(inc, dim=1)
    diff = (full - inc).abs().max().item()
    ok = diff < 1e-4
    print(f"[check-cache] KV cache 开/关最大差异 = {diff:.3e} -> {'通过' if ok else '失败'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/best.pt")
    ap.add_argument("--prompt", default="床前明月光，疑是地上霜")
    ap.add_argument("--max-new-tokens", type=int, default=50)
    ap.add_argument("--check-cache", action="store_true")
    args = ap.parse_args()

    model, tok = load_for_eval(args.ckpt)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    device = next(model.parameters()).device

    # 分词 roundtrip sanity
    assert tok.decode(tok.encode(args.prompt)) == args.prompt, "tokenizer 回环失败"

    if args.check_cache:
        if not check_cache_equivalence(model, tok):
            print("[warn] KV cache 不一致，生成结果可能不对，建议检查 src/attention.py")

    prompt_ids = tok.encode(args.prompt)
    print(f"\nprompt: {args.prompt!r}  (tokens={len(prompt_ids)})  "
          f"max_new={args.max_new_tokens}\n" + "-" * 60)
    for name, kw in PRESETS:
        t0 = time.time()
        new_ids = model.generate(prompt_ids, max_new_tokens=args.max_new_tokens, **kw)
        out = args.prompt + tok.decode(new_ids)
        print(f"\n[{name}]  ({time.time()-t0:.1f}s, {len(new_ids)} tokens)")
        print(out)
    print("\n提示：不同温度/截断策略下多样性与连贯性差异，正是 M5 想让你观察的。"
          "greedy 最稳但易重复；温度高多样但可能跑偏。")


if __name__ == "__main__":
    main()
