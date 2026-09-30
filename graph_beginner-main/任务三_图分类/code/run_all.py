# -*- coding: utf-8 -*-
"""
run_all.py —— 一键跑完所有对比实验（任务三：图分类）

用 subprocess 挨个调用 train.py，把结果全部追加写进 results/results.jsonl。
这样随时可以 Ctrl+C 中断，已经跑完的结果不会丢。

【五组实验，对应报告里要分析的内容】

    stage=main    主实验：4 个模型 × 5 个数据集（默认平均池化，分批次训练）
                  -> 回答"不同神经网络对性能的影响"

    stage=pool    池化方法对比：Avg / Max / Min × 4 个模型 × 4 个 TU 数据集
                  + ZINC 上 GCN/GIN 的池化对比
                  -> 回答"不同池化方法对图分类性能的影响"（任务三的重点）

    stage=batch   全图训练 vs 分批次训练 × 4 个模型 × 3 个数据集
                  -> 回答"全图训练和分批次训练对性能和运行时间的影响"

    stage=layers  网络层数：1/2/3/4 层 × GCN、GIN × 2 个数据集
                  -> 回答"网络层数的影响"

    stage=lr      学习率：0.001/0.005/0.01/0.05 × GCN、GIN × 2 个数据集
                  -> 回答"学习率的影响"

【怎么运行？】

    python run_all.py --stage main
    python run_all.py --stage pool
    python run_all.py --stage batch
    python run_all.py --stage layers
    python run_all.py --stage lr
    python run_all.py --stage all          # 全部跑一遍（会比较久）

    python run_all.py --stage main --dry_run     # 先看看会跑哪些命令
    python run_all.py --stage main --epochs 50   # 想快点跑就用少一点轮数
"""

import argparse
import os
import subprocess
import sys
import time

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from datasets import TU_DATASETS, ZINC_NAME   # noqa: E402
from models import MODEL_NAMES                # noqa: E402
from utils import RESULTS_PATH_HINT, format_seconds, load_results   # noqa: E402

PYTHON = sys.executable
TRAIN_PY = os.path.join(CODE_DIR, "train.py")

# 三类数据集
TU_MAIN = TU_DATASETS                      # MUTAG, PROTEINS, ENZYMES, IMDB-BINARY
ALL_MAIN = TU_DATASETS + [ZINC_NAME]        # 加上 ZINC（回归）

# 每个数据集训练多少轮
# TU 数据集很小，多跑几轮让它收敛；ZINC 一个 epoch 有几十个 batch，100 轮够了
EPOCHS = {name: 150 for name in TU_DATASETS}
EPOCHS[ZINC_NAME] = 100

# 批次大小：TU 数据集图很小，batch 用 32；ZINC 图稍大，用 64
BATCH_SIZE = {name: 32 for name in TU_DATASETS}
BATCH_SIZE[ZINC_NAME] = 64

# 池化对比实验用哪些数据集（4 个 TU 数据集 + ZINC）
POOL_DATASETS = TU_DATASETS + [ZINC_NAME]
# ZINC 数据量大、跑得慢，池化对比里只测 GCN 和 GIN 这两个最有代表性的
POOL_MODELS_ZINC = ["GCN", "GIN"]

# 做层数/学习率消融时用哪两个数据集（一个分子图、一个蛋白质图，代表性够）
ABLATION_DATASETS = ["MUTAG", "PROTEINS"]
ABLATION_MODELS = ["GCN", "GIN"]

# 全图 vs 分批 用哪些数据集（ZINC 图太多，全图训练会爆显存，所以不放进来）
BATCH_DATASETS = ["MUTAG", "PROTEINS", "IMDB-BINARY"]

# train.py 里这些超参数的默认值（生成 key 时要用同样的默认值）
DEFAULT_LR = 0.01
DEFAULT_LAYERS = 3

# 用哪些字段判断"这个实验是不是已经跑过了"
KEY_FIELDS = ("dataset", "model", "pooling", "mode", "lr", "layers", "seed", "tag")


def done_keys():
    """
    读 results/results.jsonl，返回已经跑完的实验配置集合。
    这样 run_all.py 被中断后重新运行时会自动跳过已完成的实验。
    """
    keys = set()
    for r in load_results("results.jsonl"):
        try:
            keys.add(tuple(r[f] for f in KEY_FIELDS))
        except KeyError:
            continue
    return keys


def build_experiments(stage, args):
    """按照 stage 生成实验列表。每个实验是 dict：desc + args + key。"""
    exps = []

    def add(desc, cmd_args, dataset, model, pooling, mode,
            lr=DEFAULT_LR, layers=DEFAULT_LAYERS):
        exps.append({
            "desc": desc,
            "args": cmd_args,
            # key 的顺序必须和 KEY_FIELDS 完全一致
            "key": (dataset, model, pooling, mode, lr, layers,
                    args.seed, args.tag or ""),
        })

    # ---------------- 主实验 ----------------
    if stage in ("main", "all"):
        for ds in ALL_MAIN:
            for model in MODEL_NAMES:
                add(f"[主实验] {ds:<12s} {model:<10s} pool=avg",
                    ["--dataset", ds, "--model", model,
                     "--pooling", "avg", "--mode", "sample",
                     "--epochs", str(args.epochs or EPOCHS[ds]),
                     "--batch_size", str(BATCH_SIZE[ds]),
                     "--max_graphs", str(args.max_graphs)],
                    ds, model, "avg", "sample")

    # ---------------- 池化方法对比（任务三的重点）----------------
    if stage in ("pool", "all"):
        for ds in POOL_DATASETS:
            models = POOL_MODELS_ZINC if ds == ZINC_NAME else MODEL_NAMES
            for model in models:
                for pooling in ["avg", "max", "min"]:
                    add(f"[池化]   {ds:<12s} {model:<10s} pool={pooling}",
                        ["--dataset", ds, "--model", model,
                         "--pooling", pooling, "--mode", "sample",
                         "--epochs", str(args.epochs or EPOCHS[ds]),
                         "--batch_size", str(BATCH_SIZE[ds]),
                         "--max_graphs", str(args.max_graphs)],
                        ds, model, pooling, "sample")

    # ---------------- 全图训练 vs 分批次训练 ----------------
    if stage in ("batch", "all"):
        for ds in BATCH_DATASETS:
            for model in MODEL_NAMES:
                for mode in ["full", "sample"]:
                    note = "（所有图一个批次）" if mode == "full" else ""
                    add(f"[全图vs分批] {ds:<11s} {model:<10s} {mode}{note}",
                        ["--dataset", ds, "--model", model,
                         "--pooling", "avg", "--mode", mode,
                         "--epochs", str(args.epochs or EPOCHS[ds]),
                         "--batch_size", str(BATCH_SIZE[ds])],
                        ds, model, "avg", mode)

    # ---------------- 网络层数 ----------------
    if stage in ("layers", "all"):
        for ds in ABLATION_DATASETS:
            for model in ABLATION_MODELS:
                for L in [1, 2, 3, 4]:
                    add(f"[层数]   {ds:<12s} {model:<10s} layers={L}",
                        ["--dataset", ds, "--model", model,
                         "--pooling", "avg", "--mode", "sample",
                         "--layers", str(L),
                         "--epochs", str(args.epochs or EPOCHS[ds]),
                         "--batch_size", str(BATCH_SIZE[ds])],
                        ds, model, "avg", "sample", layers=L)

    # ---------------- 学习率 ----------------
    if stage in ("lr", "all"):
        for ds in ABLATION_DATASETS:
            for model in ABLATION_MODELS:
                for lr in [0.001, 0.005, 0.01, 0.05]:
                    add(f"[学习率] {ds:<12s} {model:<10s} lr={lr}",
                        ["--dataset", ds, "--model", model,
                         "--pooling", "avg", "--mode", "sample",
                         "--lr", str(lr),
                         "--epochs", str(args.epochs or EPOCHS[ds]),
                         "--batch_size", str(BATCH_SIZE[ds])],
                        ds, model, "avg", "sample", lr=lr)

    # 所有实验共用的参数
    common = ["--quiet"]
    if args.seed is not None:
        common += ["--seed", str(args.seed)]
    if args.tag:
        common += ["--tag", args.tag]

    for e in exps:
        e["cmd"] = [PYTHON, TRAIN_PY] + e["args"] + common
    return exps


def main():
    p = argparse.ArgumentParser(description="一键跑完任务三的所有对比实验")
    p.add_argument("--stage", default="main",
                   choices=["main", "pool", "batch", "layers", "lr", "all"])
    p.add_argument("--epochs", type=int, default=0,
                   help="统一覆盖每个数据集的训练轮数（0=用推荐值）")
    p.add_argument("--max_graphs", type=int, default=5000,
                   help="ZINC 最多用多少张训练图")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--no_skip_done", action="store_true",
                   help="不跳过已经跑完的实验（默认会跳过，方便中断后接着跑）")
    args = p.parse_args()

    exps = build_experiments(args.stage, args)

    # 跳过已经跑完的实验（上一次可能被中断了）。
    # 判断依据是 results/results.jsonl 里记录的配置，只和"配置"有关，
    # 所以在哪个 stage 下跑过都能被认出来。
    n_total = len(exps)
    if not args.no_skip_done:
        done = done_keys()
        exps = [e for e in exps if e["key"] not in done]
        if n_total - len(exps) > 0:
            print(f"[跳过] 有 {n_total - len(exps)} 个实验在 results.jsonl 里已经有结果了，"
                  f"本次不重复跑。想全部重跑请加 --no_skip_done。")

    print("=" * 84)
    print(f"任务三 图分类 —— 批量实验  (stage={args.stage}, "
          f"本次要跑 {len(exps)}/{n_total} 次实验)")
    print(f"ZINC 训练图数上限={args.max_graphs}")
    print("=" * 84)

    if not exps:
        print("\n所有实验都已经跑完了，没有需要执行的。"
              "直接运行 python plot_results.py 出图即可。")
        return 0

    for i, e in enumerate(exps, 1):
        print(f"  {i:>3d}. {e['desc']}")

    if args.dry_run:
        print("\n[--dry_run] 命令示例（第 1 条）：")
        print("  " + " ".join(exps[0]["cmd"]))
        print("\n[--dry_run] 没有真的开始训练。")
        return 0

    print("\n开始执行……（结果会不断追加到 results/results.jsonl）\n")
    t_all = time.time()
    n_ok, n_fail = 0, 0
    failed = []

    for i, e in enumerate(exps, 1):
        print("-" * 84)
        print(f"[{i}/{len(exps)}] {e['desc']}")
        print("-" * 84)
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

    print("\n" + "=" * 84)
    print(f"批量实验结束：成功 {n_ok} 次，失败 {n_fail} 次，"
          f"总用时 {format_seconds(time.time() - t_all)}")
    for f in failed:
        print(f"  失败：{f}")
    print("=" * 84)
    print(RESULTS_PATH_HINT)
    print("\n下一步：运行  python plot_results.py  生成图表")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
