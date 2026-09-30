"""任务三 · 手写 LoRA（Low-Rank Adaptation），不依赖 peft。

LoRA 思想：给目标线性层 `y = Wx + b` 并排挂两条低秩矩阵 A、B，
前向叠加 `scaling * B(A x)`，其中 `scaling = alpha / r`；反向只有
A、B 可学习（原 W 置 `requires_grad=False`）。于是 Qwen2.5-0.5B 上
只训 `q_proj`/`v_proj` 两处（r=8）时，可训练参数占比约 0.1%（自检要求 < 5%）。

约定（README / tutor_prompt 检查项）：
- A: (in_features, r)，kaiming 初始化；
- B: (out_features, r)，零初始化  ->  训练一开始低秩分支输出恒为 0，
  等价于从原权重出发、不改变初始前向分布；
- forward: `y = Wx + scaling * B(A x)`；
- 反向只更新 A、B。

对外接口（eval/run.py 与 README 契约）：
- inject_lora(model, target_modules, r, alpha) -> model
- merge_lora(model) -> model

另提供 adapter 存/取（SFT/DPO 产物 = LoRA 权重目录）：
- save_adapter(model, out_dir, ...)
- load_adapter(model, adapter_dir)
- lora_state_dict(model) / count_trainable(model)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

Module = nn.Module


class LoRALinear(nn.Module):
    """把原 nn.Linear 包一层：Wx + scaling * B(A x)。

    原权重/偏置被原样接管（weight/bias 指向同一 Parameter），只新增
    lora_A、lora_B 两个参数；因此 model.parameters() 总数不变、可训
    参数只多出 A/B。
    """

    def __init__(self, linear: nn.Linear, r: int, alpha: float,
                 dropout: float = 0.0):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"LoRA 只包 nn.Linear，收到 {type(linear)}")
        if int(r) <= 0:
            raise ValueError("LoRA rank r 必须为正")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r

        # 接管原参数（同一 tensor，不复制）
        self.weight = linear.weight
        self.bias = linear.bias

        dtype, device = linear.weight.dtype, linear.weight.device
        # A: (in, r)  kaiming；B: (out, r)  全零
        self.lora_dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(
            torch.empty(self.in_features, self.r, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, self.r, dtype=dtype, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.merged = False  # merge_lora 之后为 True，分支关闭

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)          # (..., out)
        if self.merged or self.r == 0:
            return y
        z = self.lora_dropout(x) if self.training else x
        delta = (z @ self.lora_A) @ self.lora_B.t()      # (..., r) @ (r, out)
        return y + self.scaling * delta


# --------------------------------------------------------------------------
# 注入 / 合并
# --------------------------------------------------------------------------

def inject_lora(model: Module, target_modules: Union[str, Sequence[str]],
                r: int = 8, alpha: float = 16, dropout: float = 0.0,
                seed: Optional[int] = None) -> Module:
    """在名字（叶子名）属于 target_modules 的全部 nn.Linear 上注入 LoRA，
    然后把除 lora_A/lora_B 之外的所有参数冻结。

    target_modules 例：["q_proj", "v_proj"]（可匹配任意深度的同名层）。
    """
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    targets = set(target_modules)
    if seed is not None:
        torch.manual_seed(seed)

    # 先在快照里收集，避免边遍历边改树
    to_wrap = [(name, mod) for name, mod in list(model.named_modules())
               if isinstance(mod, nn.Linear)
               and not isinstance(mod, LoRALinear)
               and name.rsplit(".", 1)[-1] in targets]
    if not to_wrap:
        print(f"[lora] 警告：没有匹配到 target_modules={sorted(targets)} 的 Linear 层，"
              "请检查层名（Qwen 为 self_attn.q_proj / self_attn.v_proj）")

    for name, linear in to_wrap:
        if "." in name:
            parent_name, attr = name.rsplit(".", 1)
        else:
            parent_name, attr = "", name
        parent = model if parent_name == "" else model.get_submodule(parent_name)
        setattr(parent, attr, LoRALinear(linear, r=r, alpha=alpha,
                                         dropout=dropout))

    # 全部冻结，再单独放开 lora_A / lora_B
    for p in model.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if n.endswith(".lora_A") or n.endswith(".lora_B"):
            p.requires_grad = True

    trainable, total = count_trainable(model)
    ratio = trainable / total if total else 0.0
    print(f"[lora] 注入 {len(to_wrap)} 层 LoRA(r={r}, alpha={alpha}) | "
          f"trainable={trainable:,} / total={total:,} = {ratio:.4%}")
    return model


def merge_lora(model: Module) -> Module:
    """把 `scaling * B @ A^T` 合进原权重 W，并关闭低秩分支。

    合并后 forward 只走 W，前向输出与合并前完全一致（合并只改变实现、
    不改变数值），用于把 LoRA 权重真正落盘为一个普通模型。
    """
    merged = 0
    for m in model.modules():
        if isinstance(m, LoRALinear) and not m.merged:
            with torch.no_grad():
                # W' = W + scaling * (B @ A^T)，(out,r)@(r,in) -> (out,in)
                m.weight.add_(m.scaling * (m.lora_B @ m.lora_A.t()))
            m.merged = True
            merged += 1
    if merged:
        print(f"[lora] merge 完成：{merged} 层低秩分支已并入原权重并关闭")
    return model


# --------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------

def count_trainable(model: Module):
    """返回 (trainable_params, total_params)。"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def trainable_ratio(model: Module) -> float:
    t, total = count_trainable(model)
    return t / total if total else 0.0


# --------------------------------------------------------------------------
# adapter 存 / 取（SFT/DPO 产物目录 = adapter_config.json + adapter.bin）
# --------------------------------------------------------------------------

def lora_state_dict(model: Module) -> dict:
    """只保留各 LoRA 层的 lora_A / lora_B（不包含基座权重）。"""
    return {k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
            if k.endswith(".lora_A") or k.endswith(".lora_B")}


def save_adapter(model: Module, out_dir: Union[str, Path],
                 target_modules: Optional[Sequence[str]] = None,
                 r: Optional[int] = None,
                 alpha: Optional[float] = None,
                 meta: Optional[dict] = None) -> Path:
    """把 LoRA 权重写成一个「可独立加载」的 adapter 目录。

    ckpt/sft、ckpt/dpo 都长这样，非空目录即满足自检 `sft_vs_base`。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    st = lora_state_dict(model)
    if not st:
        raise RuntimeError("模型里没有任何 LoRA 参数（先调用 inject_lora）")
    torch.save(st, out / "adapter.bin")

    if target_modules is None or r is None or alpha is None:
        # 从已有 lora 参数名反推（key 形如 ...self_attn.q_proj.lora_A）
        leaves = set()
        for k in st:
            parts = k.split(".")
            for i, tok in enumerate(parts[:-1]):
                if parts[i + 1] in ("lora_A", "lora_B"):
                    leaves.add(tok)
        target_modules = sorted(leaves)
        # r 从 A 的第二维取（所有层一致才安全，这里取第一个）
        first_a = next(k for k in st if k.endswith(".lora_A"))
        r = int(st[first_a].shape[1])
        alpha = float(2 * r)  # 常见默认 alpha=2r；训练脚本会显式传入更准

    cfg = {"target_modules": list(target_modules), "r": int(r),
           "alpha": float(alpha), "meta": meta or {}}
    (out / "adapter_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[lora] adapter 已存 -> {out}（{len(st)} 个 lora 参数）")
    return out


def load_adapter(model: Module, adapter_dir: Union[str, Path],
                 quiet: bool = False) -> Module:
    """在已加载的基座上注入 LoRA 并载入 adapter 权重。

    要求注入目标/rank 与训练时一致——adapter_config.json 里已记录，
    因此调用方无需再传。
    """
    adir = Path(adapter_dir)
    cfg = json.loads((adir / "adapter_config.json").read_text(encoding="utf-8"))
    st = torch.load(adir / "adapter.bin", map_location="cpu")
    if not st:
        raise RuntimeError(f"{adapter_dir} 里的 adapter.bin 为空")

    inject_lora(model, cfg["target_modules"], r=cfg["r"], alpha=cfg["alpha"])
    missing, unexpected = model.load_state_dict(st, strict=False)
    if unexpected:
        raise RuntimeError(f"adapter 参数名与模型对不上：{unexpected[:5]}...")
    # load_state_dict 不改变 requires_grad 设置；inject_lora 已把非 LoRA 冻结
    if not quiet:
        n_loaded = len(st)
        print(f"[lora] 已从 {adapter_dir} 载入 {n_loaded} 个 LoRA 参数"
              f"（target={cfg['target_modules']}, r={cfg['r']}, "
              f"alpha={cfg['alpha']}）")
    return model


# --------------------------------------------------------------------------
# 便捷：从磁盘一次性建「基座 + adapter」模型
# --------------------------------------------------------------------------

def build_pretrained(base_path: str,
                     adapter_dir: Optional[Union[str, Path]] = None,
                     dtype=torch.float32, device="auto", use_fast: bool = True):
    """from_pretrained 基座（可选 dtype/device）-> 可选注入 adapter。

    返回 (model, tokenizer)。device 为 "auto" 时自动挑 cuda/mps/cpu。
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = _resolve_device(device)
    model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=dtype)
    model.to(dev)
    if adapter_dir is not None:
        load_adapter(model, adapter_dir)
        model.to(dev)  # adapter 载入后确保也在目标 device
    tok = AutoTokenizer.from_pretrained(base_path, use_fast=use_fast)
    return model, tok


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
