# -*- coding: utf-8 -*-
"""
run_all.py —— 一键跑完所有对比实验（任务四：知识图谱）

用 subprocess 挨个调用 train.py，结果全部追加写进 results/results.jsonl。

【三组实验，对应报告里要分析的内容】

    stage=main    主实验：3 个模型（TransE / RotatE / ConvE）× 2 个数据集
                  -> 回答"不同的神经网络对性能的影响"

    stage=dim     embedding 维度对比：100 / 200 / 500
                  -> 回答"不同参数对性能的影响"
                     注意：任务四的模型**没有"层数"这个超参数**
                     （TransE/RotatE 就是一个平移/旋转公式，ConvE 就是一层卷积），
                     所以这里用 embedding 维度代替"网络容量"这个维度来消融。
                     这一点在 README 里有专门说明。

    stage=lr      学习率对比：0.0001 / 0.0005 / 0.001 / 0.005
                  -> 回答"不同学习率对性能的影响"

【怎么运行？】

    python run_all.py --stage main
    python run_all.py --stage dim
    python run_all.py --stage lr
    python run_all.py --stage all

    python run_all.py --stage main --dry_run     # 先看看会跑哪些命令

【时间提示】
    知识图谱数据集比图分类大得多（FB15k-237 有 27 万条训练三元组），
    所以这一组实验整体比其他任务慢，跑之前可以先 --dry_run 看看规模。
"""

import argparse
import os
import subprocess
import sys
import time

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from datasets import DATASET_NAMES   # noqa: E402
from models import DEFAULT_DIM, MODEL_NAMES   # noqa: E402
from utils import RESULTS_PATH_HINT, format_seconds, load_results   # noqa: E402

PYTHON = sys.executable
TRAIN_PY = os.path.join(CODE_DIR, "train.py")

# 训练轮数：WN18RR 三元组少，多跑几轮；FB15k-237 有 3 倍多的数据，少跑几轮
EPOCHS = {"WN18RR": 100, "FB15k-237": 50}

# 验证集最多评估多少条。
# 为什么要限制？因为评估要"给所有实体排名"，很花时间；
# ConvE 还要把每个候选实体都和关系一起过一遍卷积，更慢。
VALID_LIMIT = {"WN18RR": 800, "FB15k-237": 800}

# 每多少轮评估一次验证集
EVAL_EVERY = 10

# 消融实验只在 WN18RR 上做（FB15k-237 太慢，跑一遍消融要几个小时）
ABLATION_DATASET = "WN18RR"

# embedding 维度候选。注意 ConvE 要求 2*dim 能被 10 整除（dim 是 5 的倍数）
DIM_CHOICES = [100, 200, 500]
LR_CHOICES = [0.0001, 0.0005, 0.001, 0.005]

# ---------------------------------------------------------------------------
# 知识图谱训练的默认超参数
#
# 【为什么学习率要显式给 0.001，而不能用通用默认值 0.01？】
#   utils.py 里给所有任务统一设的默认 lr=0.01，那是给图神经网络用的。
#   知识图谱的 embedding 是"每个实体一个自由向量"，参数之间没有权重共享，
#   lr=0.01 会让 embedding 剧烈震荡、损失根本降不下去。
#   所以这里统一显式传 --lr 0.001（这也是 KGE 论文里最常用的值）。
# ---------------------------------------------------------------------------
DEFAULT_LR = 0.001

# ---------------------------------------------------------------------------
# 【重要】权重衰减必须显式给 0，不能用通用默认值 5e-4
#
#   utils.py 给所有任务统一设的默认 weight_decay=5e-4，那是给图神经网络用的，
#   对知识图谱模型是**致命的**。实测（WN18RR + TransE + dim=100，同样跑 12 轮）：
#
#       权重衰减 = 5e-4  ->  验证 MRR = 0.0007   （基本等于没学）
#       权重衰减 = 0     ->  验证 MRR = 0.0936   （正常学习）
#
#   相差 130 倍。原因在于 Adam 的更新方式：
#       L2 正则会把 "wd × 参数" 加进梯度，而 Adam 会用梯度的二阶矩
#       把更新幅度归一化到大约 lr 这个量级。于是权重衰减这一项在 Adam 下
#       不再是"很小的收缩"，而变成了一个**和真实梯度同量级的、恒定把参数
#       往 0 拉的力**。KG 模型的参数几乎全是 embedding，被这么一直往 0 拉，
#       就学不动了。
#
#   所以 KGE 论文和主流实现（TransE / RotatE / ConvE 的官方代码、
#   OpenKE、kge_framework）里 weight_decay 一律取 0，或者小到 1e-6。
# ---------------------------------------------------------------------------
DEFAULT_WD = 0.0

# 负采样训练时的负样本个数（每条正样本抽 64 个负样本）
NUM_NEG = 64

# 损失函数。bce = 负采样二分类，三个模型通用，对比最公平。
# 不用 1n 的原因：1-N 打分要求"把所有负样本的分数一起压低"，
# 而 TransE/RotatE 是距离模型 —— 它们只要把 h+r 推向无穷远就能让所有
# 分数同时变低，损失看起来很漂亮但模型完全没学到东西（实测 MRR≈0.001）。
DEFAULT_LOSS = "bce"

# 用哪些字段判断"这个实验是不是已经跑过了"（要和 train.py 记录里的字段名一致）
# 注意 weight_decay 也在里面：万一以后有人用别的权重衰减跑过一遍，
# 那批结果的数值是不可用的，不能拿来当"已跑过"跳过。见上面 DEFAULT_WD 的说明。
KEY_FIELDS = ("dataset", "model", "dim", "lr", "loss", "num_neg",
              "weight_decay", "seed", "tag")


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

    def add(desc, dataset, model, dim, lr, extra=None):
        a = ["--dataset", dataset, "--model", model,
             "--dim", str(dim),
             "--loss", DEFAULT_LOSS,
             "--num_neg", str(NUM_NEG),
             "--lr", str(lr),
             "--weight_decay", str(DEFAULT_WD),
             "--epochs", str(args.epochs or EPOCHS[dataset]),
             "--eval_every", str(EVAL_EVERY),
             "--valid_limit", str(args.valid_limit or VALID_LIMIT[dataset])]
        if extra:
            a += extra
        exps.append({
            "desc": desc,
            "args": a,
            # key 的顺序必须和 KEY_FIELDS 完全一致
            "key": (dataset, model, dim, lr, DEFAULT_LOSS, NUM_NEG,
                    DEFAULT_WD, args.seed, args.tag or ""),
        })

    # ---------------- 主实验 ----------------
    if stage in ("main", "all"):
        for ds in DATASET_NAMES:
            for model in MODEL_NAMES:
                add(f"[主实验] {ds:<10s} {model:<7s} dim={DEFAULT_DIM[model]}",
                    ds, model, DEFAULT_DIM[model], DEFAULT_LR)

    # ---------------- embedding 维度 ----------------
    if stage in ("dim", "all"):
        for model in MODEL_NAMES:
            for dim in DIM_CHOICES:
                # ConvE 的维度约束：2*dim 必须能被 10 整除
                if model == "ConvE" and (2 * dim) % 10 != 0:
                    continue
                add(f"[维度]   {ABLATION_DATASET:<10s} {model:<7s} dim={dim}",
                    ABLATION_DATASET, model, dim, DEFAULT_LR)

    # ---------------- 学习率 ----------------
    if stage in ("lr", "all"):
        for model in MODEL_NAMES:
            for lr in LR_CHOICES:
                add(f"[学习率] {ABLATION_DATASET:<10s} {model:<7s} lr={lr}",
                    ABLATION_DATASET, model, DEFAULT_DIM[model], lr)

    common = ["--quiet"]
    if args.seed is not None:
        common += ["--seed", str(args.seed)]
    if args.tag:
        common += ["--tag", args.tag]

    for e in exps:
        e["cmd"] = [PYTHON, TRAIN_PY] + e["args"] + common
    return exps


def main():
    p = argparse.ArgumentParser(description="一键跑完任务四的所有对比实验")
    p.add_argument("--stage", default="main", choices=["main", "dim", "lr", "all"])
    p.add_argument("--epochs", type=int, default=0,
                   help="统一覆盖训练轮数（0 = 用推荐值）")
    p.add_argument("--valid_limit", type=int, default=0,
                   help="覆盖验证集评估条数上限（0 = 用推荐值）")
    p.add_argument("--seed", type=int, default=42)
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

    print("=" * 88)
    print(f"任务四 知识图谱补全 —— 批量实验  (stage={args.stage}, "
          f"本次要跑 {len(exps)}/{n_total} 次实验)")
    print(f"损失函数={DEFAULT_LOSS}  负采样数={NUM_NEG}  学习率基准={DEFAULT_LR}  "
          f"权重衰减={DEFAULT_WD}")
    print("=" * 88)

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
        print("-" * 88)
        print(f"[{i}/{len(exps)}] {e['desc']}")
        print("-" * 88)
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

    print("\n" + "=" * 88)
    print(f"批量实验结束：成功 {n_ok} 次，失败 {n_fail} 次，"
          f"总用时 {format_seconds(time.time() - t_all)}")
    for f in failed:
        print(f"  失败：{f}")
    print("=" * 88)
    print(RESULTS_PATH_HINT)
    print("\n下一步：运行  python plot_results.py  生成图表")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
