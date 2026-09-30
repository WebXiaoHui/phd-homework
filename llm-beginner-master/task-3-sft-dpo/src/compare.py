"""任务三 · 同指令对比 base / SFT / DPO 三个模型的输出（报告素材）。

用法（在 task-3-sft-dpo 目录下）：
    python src/compare.py                                        # 内置一组中英指令
    python src/compare.py --prompt "什么是 LoRA？" --prompt "翻译：深度学习很有趣"
    python src/compare.py --prompts-jsonl prompts.jsonl          # 每行 {"prompt": "..."}
    python src/compare.py --without dpo                          # SFT 还没训完时只比 base/sft

要点：三个模型一次只载入一个（用完释放），内存/显存友好。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):          # 直接 `python src/compare.py` 时补根路径
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.chat import (build_generation_text, clean_generated,
                      pad_token_id_of)
from src.lora import load_adapter

ROOT = Path(__file__).resolve().parents[1]
BASE = str(ROOT / "models" / "Qwen2.5-0.5B")
CKPT_SFT = str(ROOT / "ckpt" / "sft")
CKPT_DPO = str(ROOT / "ckpt" / "dpo")

DEFAULT_PROMPTS = [
    "什么是机器学习？请用三句话解释。",
    "把下面这句话翻译成英文：深度学习让我学到了很多有意思的知识。",
    "写一个 Python 函数，判断一个字符串是不是回文。",
    "用一句话介绍 LoRA 的原理。",
]


def load_one_model(base: str, adapter: str | None, dtype, device):
    m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype).to(device)
    if adapter is not None:
        load_adapter(m, adapter, quiet=True)
        m.to(device)
    m.eval()
    return m


@torch.no_grad()
def generate(model, tok, prompt: str, max_new_tokens: int, device: str) -> str:
    text = build_generation_text(prompt)
    ids = tok(text, return_tensors="pt").to(device)
    pad_id = pad_token_id_of(tok)
    out = model.generate(
        input_ids=ids["input_ids"],
        attention_mask=ids["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=False,                      # greedy：结果可复现、好对比
        pad_token_id=pad_id,
        eos_token_id=tok.eos_token_id,
    )
    new = out[0][ids["input_ids"].shape[1]:]
    return clean_generated(tok.decode(new, skip_special_tokens=False))


def resolve_device(arg: str):
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--sft", default=CKPT_SFT)
    ap.add_argument("--dpo", default=CKPT_DPO)
    ap.add_argument("--without", choices=["sft", "dpo"], action="append",
                    default=[])
    ap.add_argument("--prompt", action="append", default=[])
    ap.add_argument("--prompts-jsonl", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    prompts = list(args.prompt)
    if args.prompts_jsonl:
        for line in Path(args.prompts_jsonl).open(encoding="utf-8"):
            line = line.strip()
            if line:
                prompts.append(json.loads(line)["prompt"])
    prompts = prompts or DEFAULT_PROMPTS

    device = resolve_device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" and device == "cuda" \
        else torch.float32
    if args.dtype == "bf16" and device != "cuda":
        print("[warn] bf16 需要 CUDA，已回退 fp32")

    tok = AutoTokenizer.from_pretrained(args.base)
    pad_token_id_of(tok)

    variants = [("base", args.base, None)]
    if "sft" not in args.without:
        variants.append(("SFT", args.base, args.sft))
    if "dpo" not in args.without:
        variants.append(("DPO", args.base, args.dpo))

    print("=" * 72)
    print(f"对比模型：{', '.join(v[0] for v in variants)}  "
          f"device={device} dtype={dtype}")
    print("=" * 72)

    results = {}          # model -> {prompt -> text}
    for name, base, adapter in variants:
        results[name] = {}
        if adapter is not None and not Path(adapter).exists():
            print(f"\n[skip] {name} 的 adapter 目录不存在：{adapter}（跳过）")
            continue
        model = load_one_model(base, adapter, dtype, device)
        print(f"\n########## {name} ##########")
        for q in prompts:
            ans = generate(model, tok, q, args.max_new_tokens, device)
            results[name][q] = ans
            print(f"\n[指令] {q}\n[回答] {ans}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 72)
    print("小结：逐条对比同一指令下 base / SFT / DPO 的差异——")
    print("  - base 只会续写、不按指令办事；SFT 学会对话式作答；")
    print("  - DPO 相比 SFT 更倾向于给出偏好数据里的“更好”回答（更克制/更完整）。")
    print("把上面输出整段贴进提交即可作为 M4 / 报告素材。")

    # 便于复现：也落一份 json
    out = ROOT / "eval" / "compare_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n结果已存：{out}")


if __name__ == "__main__":
    main()
