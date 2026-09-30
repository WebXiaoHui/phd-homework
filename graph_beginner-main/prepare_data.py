# -*- coding: utf-8 -*-
"""
prepare_data.py —— 数据集一键下载脚本（四个任务共用）

【这个脚本是干什么的？】
    本作业需要用到很多图数据集。正常情况下，PyTorch Geometric (PyG) 会自己去
    境外的服务器下载（github / google drive / dropbox）。但是在国内网络环境下，
    这些地址大多访问不通，程序会卡住或者报 URLError。

    所以这个脚本把「下载数据」这一步单独拿出来，改用国内可访问的镜像源，
    一次性把数据放到每个任务自己的 data/ 目录里。
    之后运行训练脚本时，PyG 会发现数据已经存在，就不会再去联网下载了。

【怎么用？】
    在项目根目录（也就是本文件所在的目录）下打开终端，执行：

        # 下载全部四个任务的数据（约 1GB，第一次需要十几分钟）
        python prepare_data.py

        # 只下载某一个任务的数据
        python prepare_data.py --task 1
        python prepare_data.py --task 3

    注意：请在 conda 环境 globalwork 下运行（python 请用该环境的 python）。

【数据都存在哪里？】
    任务一_节点分类/data/   ->  Cora, Citeseer, Flickr
    任务二_链路预测/data/   ->  Cora, Citeseer, Flickr（和任务一相同的数据）
    任务三_图分类/data/     ->  MUTAG, PROTEINS, ENZYMES, IMDB-BINARY, ZINC
    任务四_知识图谱/data/   ->  WN18RR, FB15k-237

【用到了哪些镜像？】
    - jsdelivr        ：GitHub 文件加速（用来下 Planetoid 数据）
    - ghproxy.net     ：GitHub 加速（备用）
    - data.dgl.ai     ：DGL 框架的数据服务器（Flickr、WN18RR、FB15k-237）
    - hf-mirror.com   ：HuggingFace 国内镜像（ZINC）
    - chrsmrrs.com    ：图核数据集官方地址（TUDataset，国内可直连）
"""

import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile

# ---------------------------------------------------------------------------
# 【Windows 用户注意】让终端输出使用 UTF-8 编码
#
# Windows 控制台默认用 GBK 编码。当输出被重定向到文件（例如 > log.txt）时，
# Python 会改用 GBK 输出，遇到某些符号就会抛 UnicodeEncodeError 直接崩溃。
# 这里统一改成 UTF-8 并用 errors="replace" 兜底，保证不会再因为这个报错。
# ---------------------------------------------------------------------------
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 0. 基础配置
# ---------------------------------------------------------------------------

# 本文件所在的目录 = 项目根目录。用绝对路径，避免"从哪个目录运行"带来的问题。
ROOT = os.path.dirname(os.path.abspath(__file__))

# 下载时伪装成浏览器，有些服务器会拒绝 python 默认的 User-Agent
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PyG-Dataset-Downloader"}

# 每个文件的下载重试次数
MAX_RETRY = 3


def human_size(num_bytes):
    """把字节数变成人看得懂的字符串，例如 1536 -> '1.5 KB'。"""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def download(url, dst, desc=None, skip_if_exists=True, retries=MAX_RETRY):
    """
    下载一个文件到 dst。

    参数：
        url   : 下载地址
        dst   : 保存到哪个本地路径（会自动创建父目录）
        desc  : 打印日志时显示的名字
        skip_if_exists : 如果文件已存在就跳过（避免重复下载）
        retries : 失败重试次数

    返回 True 表示成功（或者本来就有），False 表示失败。
    """
    desc = desc or os.path.basename(dst)

    if skip_if_exists and os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"  [跳过] {desc} 已存在 ({human_size(os.path.getsize(dst))})")
        return True

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"  # 先下到临时文件，成功了再改名，避免下到一半中断留下坏文件

    for attempt in range(1, retries + 1):
        try:
            print(f"  [下载] {desc}  <- {url}  (第 {attempt}/{retries} 次尝试)")
            req = urllib.request.Request(url, headers=HEADERS)
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
                total = int(resp.headers.get("Content-Length", 0))
                done = 0
                while True:
                    chunk = resp.read(1024 * 256)  # 每次读 256KB
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total > 0 and done % (1024 * 1024 * 5) < 1024 * 256:
                        # 每下载大约 5MB 打印一次进度
                        print(f"         ... {human_size(done)} / {human_size(total)}")
            os.replace(tmp, dst)
            print(f"  [完成] {desc}  {human_size(os.path.getsize(dst))}  用时 {time.time()-t0:.1f}s")
            return True
        except Exception as e:
            print(f"  [失败] {desc}: {type(e).__name__}: {e}")
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt < retries:
                time.sleep(3)

    return False


def download_first_ok(urls, dst, desc=None):
    """依次尝试多个镜像地址，只要有一个成功就行。"""
    for i, url in enumerate(urls):
        if download(url, dst, desc=desc if i == 0 else f"{desc} (镜像{i+1})"):
            return True
    print(f"  [错误] {desc} 所有镜像都下载失败：{urls}")
    return False


def unzip_to(zip_path, out_dir):
    """解压 zip 到指定目录。"""
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)


# ---------------------------------------------------------------------------
# 1. 任务一 / 任务二：Cora、Citeseer、Flickr
# ---------------------------------------------------------------------------

# Planetoid 数据集（Cora / Citeseer）需要的 8 个原始文件。
# 这些文件的命名是 PyG 写死的，放在 <root>/<数据集名>/raw/ 下，PyG 就不会再联网下载。
PLANETOID_FILES = ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]

# 镜像源（国内可访问）。jsdelivr 是主力，ghproxy 作为备用。
PLANETOID_MIRRORS = [
    "https://cdn.jsdelivr.net/gh/kimiyoung/planetoid@master/data/{f}",
    "https://ghproxy.net/https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/{f}",
]

# Flickr 原始数据（GraphSAINT 版本，89250 个节点）。
# data.dgl.ai 上打包好了 4 个文件，正好和 PyG 需要的一模一样。
FLICKR_URL = "https://data.dgl.ai/dataset/flickr.zip"
FLICKR_FILES = ["adj_full.npz", "feats.npy", "class_map.json", "role.json"]


def prepare_planetoid(data_root, name):
    """下载 Cora 或 Citeseer 的原始文件到 <data_root>/<name>/raw/。"""
    print(f"\n--- 准备 Planetoid 数据集：{name} ---")
    raw_dir = os.path.join(data_root, name, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    lower = name.lower()
    ok = True
    for suffix in PLANETOID_FILES:
        fname = f"ind.{lower}.{suffix}"
        urls = [m.format(f=fname) for m in PLANETOID_MIRRORS]
        ok &= download_first_ok(urls, os.path.join(raw_dir, fname), desc=fname)
    return ok


def prepare_flickr(data_root):
    """下载 Flickr：先下 dgl 的 zip，再把里面 4 个文件放进 PyG 期望的目录。"""
    print("\n--- 准备 Flickr 数据集 ---")
    raw_dir = os.path.join(data_root, "Flickr", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # 4 个文件都齐了就直接返回
    if all(os.path.exists(os.path.join(raw_dir, f)) for f in FLICKR_FILES):
        print("  [跳过] Flickr 原始文件已齐全")
        return True

    zip_path = os.path.join(ROOT, "_download_cache", "flickr.zip")
    if not download(FLICKR_URL, zip_path, desc="flickr.zip"):
        return False

    print("  [解压] flickr.zip -> Flickr/raw/")
    unzip_to(zip_path, raw_dir)

    # 有些压缩包里多套了一层目录，这里做一下兼容处理
    for f in FLICKR_FILES:
        if not os.path.exists(os.path.join(raw_dir, f)):
            for dirpath, _, files in os.walk(raw_dir):
                if f in files:
                    shutil.copy(os.path.join(dirpath, f), os.path.join(raw_dir, f))
                    break

    missing = [f for f in FLICKR_FILES if not os.path.exists(os.path.join(raw_dir, f))]
    if missing:
        print(f"  [错误] Flickr 缺少文件：{missing}")
        return False
    print("  [完成] Flickr 原始文件准备完毕")
    return True


def prepare_task12():
    """
    任务一（节点分类）和任务二（链路预测）用的是同一批数据集：
    Cora / Citeseer / Flickr。
    """
    print("=" * 78)
    print("准备【任务一 节点分类】和【任务二 链路预测】的数据集")
    print("=" * 78)
    ok = True
    for task_dir in ["任务一_节点分类", "任务二_链路预测"]:
        data_root = os.path.join(ROOT, task_dir, "data")
        print(f"\n########## 下载到 {task_dir}/data/ ##########")
        for name in ["Cora", "Citeseer"]:
            ok &= prepare_planetoid(data_root, name)
        ok &= prepare_flickr(data_root)
    return ok


# ---------------------------------------------------------------------------
# 2. 任务三：TUDataset（MUTAG/PROTEINS/...）和 ZINC
# ---------------------------------------------------------------------------

# TUDataset 官方地址，国内可以直接访问，不需要镜像
TU_URL = "https://www.chrsmrrs.com/graphkerneldatasets/{name}.zip"

# 我们选用这几个规模较小、跑得快的图分类数据集
TU_NAMES = ["MUTAG", "PROTEINS", "ENZYMES", "IMDB-BINARY"]

# ZINC 在 HuggingFace 镜像上的地址（原版在 dropbox 上，国内下不动）
# 每个文件是一行一个 JSON，字段为：node_feat / edge_index / edge_attr / y / num_nodes
ZINC_URL = "https://hf-mirror.com/datasets/graphs-datasets/ZINC/resolve/main/{split}.jsonl"
ZINC_SPLITS = ["train", "val", "test"]


def prepare_tu(data_root):
    """下载 TUDataset：PyG 期望 <root>/<name>/raw/<name>.zip 和 <name>_*.txt。"""
    print("\n--- 准备 TUDataset ---")
    ok = True
    for name in TU_NAMES:
        raw_dir = os.path.join(data_root, name, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        zip_path = os.path.join(raw_dir, f"{name}.zip")

        if not download(TU_URL.format(name=name), zip_path, desc=f"{name}.zip"):
            ok = False
            continue

        # PyG 的 TUDataset 在 process() 阶段需要解压后的 txt 文件。
        # 我们先解压好，这样 PyG 的 download() 就不会被触发。
        try:
            unzip_to(zip_path, raw_dir)
        except Exception as e:
            print(f"  [警告] {name} 解压失败: {e}")
            ok = False
    return ok


def prepare_zinc(data_root):
    """
    下载 ZINC 的 jsonl 文件。

    说明：PyG 自带的 ZINC 类是从 dropbox 下 molecules.zip，国内访问不了。
    所以我们改从 HuggingFace 镜像拿同样内容的数据（已经转成 jsonl）。
    任务三的代码里有一个自定义的 ZINCDataset 类负责读取这些 jsonl。
    """
    print("\n--- 准备 ZINC 数据集（从 HuggingFace 镜像）---")
    raw_dir = os.path.join(data_root, "ZINC", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    ok = True
    for split in ZINC_SPLITS:
        ok &= download(ZINC_URL.format(split=split),
                       os.path.join(raw_dir, f"{split}.jsonl"),
                       desc=f"ZINC/{split}.jsonl")
    return ok


def prepare_task3():
    """任务三（图分类）的数据集。"""
    print("=" * 78)
    print("准备【任务三 图分类】的数据集")
    print("=" * 78)
    data_root = os.path.join(ROOT, "任务三_图分类", "data")
    ok = prepare_tu(data_root)
    ok &= prepare_zinc(data_root)
    return ok


# ---------------------------------------------------------------------------
# 3. 任务四：知识图谱数据集（WN18RR / FB15k-237）
# ---------------------------------------------------------------------------

KG_URL = "https://data.dgl.ai/dataset/{name}.zip"
KG_NAMES = ["WN18RR", "FB15k-237"]


def prepare_task4():
    """
    任务四（知识图谱补全）的数据集。

    zip 里是标准的三元组文本文件：train.txt / valid.txt / test.txt
    每行格式：头实体 <TAB> 关系 <TAB> 尾实体
    """
    print("=" * 78)
    print("准备【任务四 知识图谱】的数据集")
    print("=" * 78)
    data_root = os.path.join(ROOT, "任务四_知识图谱", "data")
    cache = os.path.join(ROOT, "_download_cache")
    os.makedirs(cache, exist_ok=True)
    ok = True

    for name in KG_NAMES:
        print(f"\n--- 准备 {name} ---")
        out_dir = os.path.join(data_root, name)
        # 三个文件都在就跳过
        if all(os.path.exists(os.path.join(out_dir, f)) for f in ["train.txt", "valid.txt", "test.txt"]):
            print(f"  [跳过] {name} 已存在")
            continue

        zip_path = os.path.join(cache, f"{name}.zip")
        if not download(KG_URL.format(name=name), zip_path, desc=f"{name}.zip"):
            ok = False
            continue
        try:
            unzip_to(zip_path, out_dir)
        except Exception as e:
            print(f"  [错误] {name} 解压失败: {e}")
            ok = False
            continue

        # 压缩包里可能多一层同名目录，把文件提到 out_dir 根下
        for f in ["train.txt", "valid.txt", "test.txt"]:
            if not os.path.exists(os.path.join(out_dir, f)):
                for dirpath, _, files in os.walk(out_dir):
                    if f in files:
                        shutil.copy(os.path.join(dirpath, f), os.path.join(out_dir, f))
                        break
        print(f"  [完成] {name}")
    return ok


# ---------------------------------------------------------------------------
# 4. 主入口
# ---------------------------------------------------------------------------

TASKS = {
    "1": ("任务一 节点分类", prepare_task12),
    "2": ("任务二 链路预测", prepare_task12),   # 和任务一用同一批数据
    "3": ("任务三 图分类", prepare_task3),
    "4": ("任务四 知识图谱", prepare_task4),
}


def main():
    parser = argparse.ArgumentParser(description="下载本作业所需的全部图数据集")
    parser.add_argument("--task", default="all",
                        help="要准备哪个任务的数据：1 / 2 / 3 / 4 / all（默认 all）")
    args = parser.parse_args()

    print(__doc__)
    print(f"项目根目录：{ROOT}\n")

    if args.task == "all":
        todo = ["1", "3", "4"]      # 任务 1 和 2 一起做，不用重复
    else:
        todo = [args.task]

    results = {}
    for t in todo:
        label, fn = TASKS[t]
        results[label] = fn()

    print("\n" + "=" * 78)
    print("下载结果汇总")
    print("=" * 78)
    for label, ok in results.items():
        print(f"  {'[成功]' if ok else '[失败]'}  {label}")
    print("\n提示：如果某个数据集失败，多半是网络抖动，重新运行本脚本即可（已下载的会跳过）。")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
