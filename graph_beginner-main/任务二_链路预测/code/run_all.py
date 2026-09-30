# -*- coding: utf-8 -*-
"""
run_all.py —— 一键跑完所有对比实验（任务二：链路预测）

三组实验，和任务一的结构保持一致，方便对照：

    stage=main    4 个模型 × 3 个数据集 × 2 种训练方式 = 24 次
    stage=lr      学习率影响：[0.001, 0.005, 0.01, 0.05]
    stage=layers  层数影响：  [2, 3, 4]

【怎么运行？】
    python run_all.py --stage main
    python run_all.py --stage lr
    python run_all.py --stage layers
    python run_all.py --stage all          # 全部

    python run_all.py --stage main --dry_run     # 只看看会跑什么

【小提示】
    - 结果追加写入 results/results.jsonl，可以随时 Ctrl+C 中断。
    - 想跑快一点：python run_all.py --stage main --epochs 50
"""

import argparse
import os
import subprocess
import sys
import time

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from utils import RESULTS_PATH_HINT, format_seconds, load_results   # noqa: E402

PYTHON = sys.executable
TRAIN_PY = os.path.join(CODE_DIR, "train.py")

DATASETS = ["Cora", "Citeseer", "Flickr"]
MODELS = ["GCN", "GAT", "GraphSAGE", "GIN"]
MODES = ["full", "sample"]

# 采样训练用的 batch_size（每批种子节点数）
# 小图用小 batch，让每个 epoch 有多次参数更新；大图用大 batch 控制训练时间
BATCH_SIZE = {"Cora": 256, "Citeseer": 256, "Flickr": 4096}

# 评估比较费时（要算几百万条负样本的分数），小图每轮都评估，
# 大图每 5 轮评估一次，最后第 1 轮和最后 1 轮一定会评估
EVAL_EVERY = {"Cora": 1, "Citeseer": 1, "Flickr": 5}

# train.py 里这些超参数的默认值（生成 key 时要用同样的默认值）
DEFAULT_LR = 0.01
DEFAULT_LAYERS = 2

# 用哪些字段判断"这个实验是不是已经跑过了"
KEY_FIELDS = ("dataset", "model", "mode", "lr", "layers", "seed", "tag")


def done_keys():
    """读 results/results.jsonl，返回已经跑完的实验配置集合（方便中断后接着跑）。"""
    keys = set()
    for r in load_results("results.jsonl"):
        try:
            keys.add(tuple(r[f] for f in KEY_FIELDS))
        except KeyError:
            continue
    return keys


def build_experiments(stage, args):
    exps = []

    def add(desc, cmd_args, dataset, model, mode,
            lr=DEFAULT_LR, layers=DEFAULT_LAYERS):
        exps.append({
            "desc": desc,
            "args": cmd_args,
            # key 的顺序必须和 KEY_FIELDS 完全一致
            "key": (dataset, model, mode, lr, layers, args.seed, args.tag or ""),
        })

    if stage in ("main", "all"):
        for ds in DATASETS:
            for model in MODELS:
                for mode in MODES:
                    a = ["--dataset", ds, "--model", model, "--mode", mode,
                         "--eval_every", str(EVAL_EVERY[ds])]
                    extra = ""
                    if mode == "sample":
                        a += ["--batch_size", str(BATCH_SIZE[ds])]
                        extra = f" (bs={BATCH_SIZE[ds]})"
                    add(f"[主实验] {ds:<9s} {model:<10s} {mode}{extra}",
                        a, ds, model, mode)

    if stage in ("lr", "all"):
        for ds in ["Cora", "Flickr"]:
            for lr in [0.001, 0.005, 0.01, 0.05]:
                add(f"[学习率] {ds:<9s} GCN full lr={lr}",
                    ["--dataset", ds, "--model", "GCN", "--mode", "full",
                     "--lr", str(lr), "--eval_every", str(EVAL_EVERY[ds])],
                    ds, "GCN", "full", lr=lr)

    if stage in ("layers", "all"):
        for ds in ["Cora", "Flickr"]:
            for L in [2, 3, 4]:
                add(f"[层数]   {ds:<9s} GCN full layers={L}",
                    ["--dataset", ds, "--model", "GCN", "--mode", "full",
                     "--layers", str(L), "--eval_every", str(EVAL_EVERY[ds])],
                    ds, "GCN", "full", layers=L)

    common = ["--epochs", str(args.epochs), "--quiet"]
    if args.seed is not None:
        common += ["--seed", str(args.seed)]
    if args.tag:
        common += ["--tag", args.tag]
    if args.decoder:
        common += ["--decoder", args.decoder]

    for e in exps:
        e["cmd"] = [PYTHON, TRAIN_PY] + e["args"] + common
    return exps


def main():
    p = argparse.ArgumentParser(description="一键跑完任务二的所有对比实验")
    p.add_argument("--stage", default="main",
                   choices=["main", "lr", "layers", "all"])
    p.add_argument("--epochs", type=int, default=100, help="每次实验训练多少轮")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--decoder", type=str, default="dot", choices=["dot", "mlp"],
                   help="解码器类型")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--no_skip_done", action="store_true",
                   help="不跳过已经跑完的实验（默认会跳过，方便中断后接着跑）")
    args = p.parse_args()

    exps = build_experiments(args.stage, args)

    # 跳过已经跑完的实验（上一次可能被中断了）
    n_total = len(exps)
    if not args.no_skip_done:
        done = done_keys()
        exps = [e for e in exps if e["key"] not in done]
        if n_total - len(exps) > 0:
            print(f"[跳过] 有 {n_total - len(exps)} 个实验在 results.jsonl 里已经有结果了，"
                  f"本次不重复跑。想全部重跑请加 --no_skip_done。")

    print("=" * 78)
    print(f"任务二 链路预测 —— 批量实验  (stage={args.stage}, "
          f"本次要跑 {len(exps)}/{n_total} 次实验)")
    print(f"每次训练 {args.epochs} 轮 | 解码器={args.decoder}")
    print("=" * 78)

    if not exps:
        print("\n所有实验都已经跑完了，没有需要执行的。"
              "直接运行 python plot_results.py 出图即可。")
        return 0

    for i, e in enumerate(exps, 1):
        print(f"  {i:>3d}. {e['desc']}")

    if args.dry_run:
        print("\n[--dry_run] 命令示例：")
        print("  " + " ".join(exps[0]["cmd"]))
        return 0

    print("\n开始执行……\n")
    t_all = time.time()
    n_ok, n_fail, failed = 0, 0, []

    for i, e in enumerate(exps, 1):
        print("-" * 78)
        print(f"[{i}/{len(exps)}] {e['desc']}")
        print("-" * 78)
        t0 = time.time()
        proc = subprocess.run(e["cmd"], cwd=CODE_DIR)
        dt = time.time() - t0
        if proc.returncode == 0:
            n_ok += 1
            print(f"  -> 完成，用时 {format_seconds(dt)}")
        else:
            n_fail += 1
            failed.append(e["desc"])
            print(f"  -> 【失败】返回码 {proc.returncode}，用时 {format_seconds(dt)}")

    print("\n" + "=" * 78)
    print(f"批量实验结束：成功 {n_ok} 次，失败 {n_fail} 次，"
          f"总用时 {format_seconds(time.time() - t_all)}")
    for f in failed:
        print(f"  失败：{f}")
    print("=" * 78)
    print(RESULTS_PATH_HINT)
    print("\n下一步：运行  python plot_results.py  生成图表")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
