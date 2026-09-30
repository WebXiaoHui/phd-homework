# -*- coding: utf-8 -*-
"""
run_all.py —— 一键跑完所有对比实验（任务一：节点分类）

【这个脚本解决什么问题？】
    作业要求做很多组对比实验，手动一条条敲命令太累也容易漏。这个脚本把
    所有实验列成一张"清单"，然后逐个调用 train.py 跑完，最后打印一张汇总表。

【三组实验】

    stage=main   （主实验，必做）
        4 个模型 × 3 个数据集 × 2 种训练方式 = 24 次实验
        回答的问题：哪个模型好？哪种数据集难？全图训练和采样训练谁更快、谁更准？

    stage=lr     （学习率的影响，报告要求）
        在 Cora 和 Flickr 上，把学习率取 [0.001, 0.005, 0.01, 0.05]
        回答的问题：学习率对性能影响有多大？太大/太小分别会怎样？

    stage=layers （网络层数的影响，报告要求）
        层数取 [2, 3, 4]
        回答的问题：层数越多越好吗？为什么会出现"过平滑"？

【怎么运行？】

    # 先跑主实验（大约 20-40 分钟，取决于显卡）
    python run_all.py --stage main

    # 再跑学习率和层数的消融实验
    python run_all.py --stage lr
    python run_all.py --stage layers

    # 全部一起跑
    python run_all.py --stage all

    # 只想看看会执行哪些命令（不真的跑）
    python run_all.py --stage main --dry_run

【小提示】
    - 每一次实验都会往 results/results.jsonl 追加一行结果，可以随时中断。
    - 重复运行会出现重复记录。想重新开始，把 results/results.jsonl 删掉即可。
    - 想快速验证流程是否通顺，可以加 --epochs 10。
"""

import argparse
import os
import subprocess
import sys
import time

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from utils import RESULTS_PATH_HINT, format_seconds, load_results  # noqa: E402

PYTHON = sys.executable          # 用当前这个解释器去跑 train.py，保证环境一致
TRAIN_PY = os.path.join(CODE_DIR, "train.py")

DATASETS = ["Cora", "Citeseer", "Flickr"]
MODELS = ["GCN", "GAT", "GraphSAGE", "GIN"]
MODES = ["full", "sample"]

# 采样训练时的 batch_size（每批种子节点数）。
# 为什么不同数据集用不一样的值？
#   Cora/Citeseer 一共才两三千个节点，batch 太大会退化成"全图训练"，
#   失去采样的意义，所以用 128 左右，让每个 epoch 有若干次参数更新。
#   Flickr 有 8.9 万个节点，batch 太小会导致每个 epoch 的 batch 数太多、训练很慢，
#   所以用 4096。
BATCH_SIZE = {"Cora": 128, "Citeseer": 128, "Flickr": 4096}

# train.py 里这些参数的默认值。生成实验时要按同样的默认值填进 key，
# 才能和结果文件里记录的值对得上。
DEFAULT_LR = 0.01
DEFAULT_LAYERS = 2

# 用哪些字段来判断"这个实验是不是已经跑过了"
KEY_FIELDS = ("dataset", "model", "mode", "lr", "layers", "seed", "tag")


def done_keys():
    """
    读 results/results.jsonl，返回"已经跑完的实验"的配置集合。

    这样重复运行  run_all.py  时（比如上一次中途被 Ctrl+C 了），
    已经完成的实验会被自动跳过，不用重跑。用 --no_skip_done 可以关掉这个行为。
    """
    keys = set()
    for r in load_results("results.jsonl"):
        try:
            keys.add(tuple(r[f] for f in KEY_FIELDS))
        except KeyError:
            continue
    return keys


def build_experiments(stage: str, args):
    """
    根据 stage 生成实验清单。每一项是一个字典：
        {"desc": 给人看的名字, "args": 传给 train.py 的参数列表,
         "key":  用来判断是否已跑过的配置元组}
    """
    exps = []

    def add(desc, cmd_args, dataset, model, mode,
            lr=DEFAULT_LR, layers=DEFAULT_LAYERS):
        """把一条实验加进清单，同时算好它的 key。"""
        exps.append({
            "desc": desc,
            "args": cmd_args,
            # key 的顺序必须和 KEY_FIELDS 完全一致
            "key": (dataset, model, mode, lr, layers,
                    args.seed, args.tag or ""),
        })

    if stage in ("main", "all"):
        # ---- 主实验：模型 × 数据集 × 训练方式 ----
        for ds in DATASETS:
            for model in MODELS:
                for mode in MODES:
                    cmd_args = ["--dataset", ds, "--model", model, "--mode", mode]
                    if mode == "sample":
                        cmd_args += ["--batch_size", str(BATCH_SIZE[ds])]
                    add(f"[主实验] {ds:<9s} {model:<10s} {mode}"
                        + (f" (bs={BATCH_SIZE[ds]})" if mode == "sample" else ""),
                        cmd_args, ds, model, mode)

    if stage in ("lr", "all"):
        # ---- 学习率消融：固定 GCN，只改 lr ----
        for ds in ["Cora", "Flickr"]:
            for lr in [0.001, 0.005, 0.01, 0.05]:
                add(f"[学习率] {ds:<9s} GCN full lr={lr}",
                    ["--dataset", ds, "--model", "GCN", "--mode", "full",
                     "--lr", str(lr)],
                    ds, "GCN", "full", lr=lr)

    if stage in ("layers", "all"):
        # ---- 层数消融：固定 GCN，只改层数 ----
        for ds in ["Cora", "Flickr"]:
            for L in [2, 3, 4]:
                add(f"[层数]   {ds:<9s} GCN full layers={L}",
                    ["--dataset", ds, "--model", "GCN", "--mode", "full",
                     "--layers", str(L)],
                    ds, "GCN", "full", layers=L)

    # 把公共参数拼到每一条命令后面
    common = ["--epochs", str(args.epochs), "--quiet"]
    if args.seed is not None:
        common += ["--seed", str(args.seed)]
    if args.tag:
        common += ["--tag", args.tag]

    for e in exps:
        e["cmd"] = [PYTHON, TRAIN_PY] + e["args"] + common

    return exps


def main():
    parser = argparse.ArgumentParser(description="一键跑完任务一的所有对比实验")
    parser.add_argument("--stage", default="main",
                        choices=["main", "lr", "layers", "all"],
                        help="跑哪一组实验")
    parser.add_argument("--epochs", type=int, default=200, help="每次实验训练多少轮")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--tag", type=str, default="", help="给这批实验加标签")
    parser.add_argument("--dry_run", action="store_true", help="只打印命令，不真的执行")
    parser.add_argument("--keep_going", action="store_true", default=True,
                        help="某次实验失败也继续往下跑（默认开启）")
    parser.add_argument("--no_skip_done", action="store_true",
                        help="不跳过已经跑完的实验（默认会跳过，方便中断后续跑）")
    args = parser.parse_args()

    exps = build_experiments(args.stage, args)

    # 跳过已经跑完的实验（上一次可能被 Ctrl+C 中断了）
    n_total = len(exps)
    if not args.no_skip_done:
        done = done_keys()
        exps = [e for e in exps if e["key"] not in done]
        if n_total - len(exps) > 0:
            print(f"[跳过] 有 {n_total - len(exps)} 个实验在 results.jsonl 里已经有结果了，"
                  f"本次不重复跑。想全部重跑请加 --no_skip_done。")

    print("=" * 78)
    print(f"任务一 节点分类 —— 批量实验  (stage={args.stage}, "
          f"本次要跑 {len(exps)}/{n_total} 次实验)")
    print(f"每次训练 {args.epochs} 轮 | 结果会写入 results/results.jsonl")
    print("=" * 78)

    if not exps:
        print("\n所有实验都已经跑完了，没有需要执行的。直接运行 "
              "python plot_results.py 出图即可。")
        return 0

    for i, e in enumerate(exps, 1):
        print(f"  {i:>3d}. {e['desc']}")

    if args.dry_run:
        print("\n[--dry_run] 只打印命令，不执行。完整命令示例：")
        print("  " + " ".join(exps[0]["cmd"]))
        return 0

    print("\n开始执行……（可以按 Ctrl+C 中断，已完成的结果会保留）\n")

    t_all = time.time()
    n_ok, n_fail = 0, 0
    failed = []

    for i, e in enumerate(exps, 1):
        print("-" * 78)
        print(f"[{i}/{len(exps)}] {e['desc']}")
        print("-" * 78)
        t0 = time.time()

        # subprocess.run 会等 train.py 跑完；一次实验一个独立进程，
        # 好处是显存能完全释放，不会因为上一个模型没清干净影响下一个。
        proc = subprocess.run(e["cmd"], cwd=CODE_DIR)

        dt = time.time() - t0
        if proc.returncode == 0:
            n_ok += 1
            print(f"  -> 完成，用时 {format_seconds(dt)}")
        else:
            n_fail += 1
            failed.append(e["desc"])
            print(f"  -> 【失败】返回码 {proc.returncode}，用时 {format_seconds(dt)}")
            if not args.keep_going:
                break

    print("\n" + "=" * 78)
    print(f"批量实验结束：成功 {n_ok} 次，失败 {n_fail} 次，"
          f"总用时 {format_seconds(time.time() - t_all)}")
    if failed:
        print("失败的实验：")
        for f in failed:
            print(f"  - {f}")
    print("=" * 78)
    print(RESULTS_PATH_HINT)
    print("\n下一步：运行  python plot_results.py  生成图表")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
