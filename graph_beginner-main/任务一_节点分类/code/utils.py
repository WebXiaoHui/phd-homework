# -*- coding: utf-8 -*-
"""
utils.py —— 通用小工具（任务一：节点分类）

这里放的都是"和具体模型无关"的辅助功能，包括：
    1. 固定随机种子（让实验可复现）
    2. 选择计算设备（有 GPU 就用 GPU）
    3. 计时器
    4. 同时往终端和文件写日志的工具
    5. 保存实验结果的工具
    6. 命令行参数的公共部分（所有训练脚本共用同一套超参数）

新手提示：
    这个文件不直接运行，它是被 train.py / run_all.py 等脚本 import 的。
"""

import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 【Windows 用户注意】让终端输出使用 UTF-8 编码
#
# 背景：Windows 的控制台默认用 GBK 编码。当输出被重定向到文件或用管道时，
#       Python 会用 GBK 编码输出，遇到一些生僻字符就会抛 UnicodeEncodeError。
#       这里统一改成 UTF-8，并且用 errors="replace" 兜底，保证绝对不会崩。
#       （在真正的 Windows 控制台窗口里，Python 本来就是用 UTF-8 的，这句是保险措施）
# ---------------------------------------------------------------------------
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# 路径：以本文件为基准往上找，这样不管你在哪个目录执行脚本都能找到正确的位置
# ---------------------------------------------------------------------------
CODE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../任务一_节点分类/code
TASK_DIR = os.path.dirname(CODE_DIR)                    # .../任务一_节点分类
DATA_DIR = os.path.join(TASK_DIR, "data")               # .../任务一_节点分类/data
LOG_DIR = os.path.join(TASK_DIR, "logs")                # .../任务一_节点分类/logs
RESULT_DIR = os.path.join(TASK_DIR, "results")          # .../任务一_节点分类/results

# 一条给用户看的提示信息，说明结果文件在哪
RESULTS_PATH_HINT = f"实验结果保存在：{os.path.join(RESULT_DIR, 'results.jsonl')}"

for d in (DATA_DIR, LOG_DIR, RESULT_DIR):
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. 随机种子
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42):
    """
    固定所有随机源，保证同一个命令跑两次得到一样的结果。

    为什么要做这件事？
        神经网络初始化、Dropout、负采样、数据打乱……都带随机性。
        如果不固定种子，两次实验的差别可能只是"运气不同"，而不是"模型不同"。
        做对比实验时，这一点非常重要。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 下面这行让 cuDNN 只挑"确定的"算法。速度可能略慢，但结果可复现。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 2. 设备
# ---------------------------------------------------------------------------
def get_device(verbose: bool = True):
    """自动选择设备：能用 GPU 就用 GPU，否则用 CPU。"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"[设备] 使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if verbose:
            print("[设备] 未检测到可用 GPU，使用 CPU")
    return device


# ---------------------------------------------------------------------------
# 3. 计时器
# ---------------------------------------------------------------------------
class Timer:
    """
    一个简单的计时器，支持 with 语法：

        t = Timer()
        with t:
            ...干点活...
        print(t.elapsed)      # 刚才那段时间，单位秒
    """

    def __init__(self):
        self.elapsed = 0.0
        self._t0 = None

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self._t0
        return False

    @staticmethod
    def now():
        return time.time()


# ---------------------------------------------------------------------------
# 4. 日志
# ---------------------------------------------------------------------------
def setup_logger(name: str, log_file: str = None, quiet: bool = False) -> logging.Logger:
    """
    创建一个 logger：既打印到屏幕，也写到文件（方便之后交作业时附上运行日志）。

    参数：
        name     : 日志器名字，一般用实验名
        log_file : 日志文件路径。传 None 就只打印到屏幕。
        quiet    : True 表示不打印到屏幕（批量跑实验时用，避免刷屏）
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()          # 防止重复添加 handler 导致日志重复
    logger.propagate = False

    fmt = logging.Formatter("%(message)s")

    if not quiet:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        # encoding='utf-8' 很重要：日志里有中文，Windows 默认编码会乱码
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# 5. 结果保存
# ---------------------------------------------------------------------------
def save_result(record: dict, filename: str = "results.jsonl"):
    """
    把一次实验的结果追加保存成一行 JSON（jsonl 格式）。

    为什么用 jsonl？
        每次实验追加一行，随时可以中断，不会因为程序崩溃丢掉之前的结果。
        后面的 plot_results.py 会读这个文件来画图。
    """
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, filename)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def load_results(filename: str = "results.jsonl"):
    """读取 results.jsonl，返回一个字典列表。"""
    path = os.path.join(RESULT_DIR, filename)
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# 6. 公共命令行参数
# ---------------------------------------------------------------------------
def add_common_args(parser):
    """
    给 argparse 的 parser 加上所有训练脚本都需要的公共参数。
    这样 train.py / run_all.py 可以共用一套参数定义。
    """
    g = parser.add_argument_group("通用参数")
    g.add_argument("--seed", type=int, default=42, help="随机种子")
    g.add_argument("--hidden", type=int, default=128, help="隐藏层维度")
    g.add_argument("--layers", type=int, default=2, help="GNN 层数")
    g.add_argument("--dropout", type=float, default=0.5, help="Dropout 概率")
    g.add_argument("--lr", type=float, default=0.01, help="学习率")
    g.add_argument("--weight_decay", type=float, default=5e-4, help="权重衰减（L2 正则）")
    g.add_argument("--epochs", type=int, default=200, help="训练轮数")
    g.add_argument("--device", type=str, default="auto", help="auto / cpu / cuda")
    g.add_argument("--log", type=str, default=None, help="日志文件名（相对于 logs/ 目录）")
    g.add_argument("--result_file", type=str, default="results.jsonl",
                   help="结果写入 results/ 下的哪个文件")
    g.add_argument("--quiet", action="store_true", help="不往屏幕打印详细日志")
    g.add_argument("--tag", type=str, default="", help="给这次实验加个备注标签")
    return parser


def resolve_device(arg: str):
    """把 --device 参数变成真正的 torch.device 对象。"""
    if arg == "auto":
        return get_device()
    return torch.device(arg)


def count_parameters(model) -> int:
    """统计模型有多少个可训练参数（用来对比模型大小）。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_seconds(s: float) -> str:
    """把秒数格式化成 '1m23.4s' 这种好读的形式。"""
    if s < 60:
        return f"{s:.2f}s"
    m, sec = divmod(s, 60)
    return f"{int(m)}m{sec:.1f}s"
