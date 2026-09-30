# -*- coding: utf-8 -*-
"""
datasets.py —— 图分类数据集（任务三：图分类）

【图分类和前面两个任务有什么不同？】

    任务一（节点分类）：一张大图，给每个**节点**打标签
    任务二（链路预测）：一张大图，判断**边**存不存在
    任务三（图分类）  ：一大堆**小图**，给每张**图**打一个标签

    比如 MUTAG 数据集：188 个分子，每个分子是一张小图，
    节点是原子、边是化学键，标签是"这个分子有没有致突变性"。
    所以这里的"样本"是一整张图。

【本任务用到的数据集】

    TUDataset 系列（都是图分类任务，用准确率衡量）：
        MUTAG       : 188 个分子图，2 类（有没有致突变性）
        PROTEINS    : 1113 个蛋白质图，2 类（是不是酶）
        ENZYMES     : 600 个蛋白质图，6 类（属于哪类酶）
        IMDB-BINARY : 1000 个演员合作网络，2 类（电影类型）
                      注意：这个数据集**没有节点特征**，只有结构，
                      所以下面会自动用"节点度数"做 one-hot 当特征。

    ZINC（图回归任务，用 MAE 衡量，MAE 越小越好）：
        25 万个分子图，标签是一个化学性质（logP_SA_cycle_normalized）
        注意这不是分类而是**回归**！所以损失函数和评估指标都不一样。

【训练集/验证集/测试集怎么划分？】

    - TUDataset 系列：官方没有给划分，本代码用**分层随机划分**
      （stratified split，80% / 10% / 10%）。
      为什么要"分层"？因为 MUTAG 只有 188 个样本，如果随机切，
      有可能某个类别在训练集里一个都没有。分层划分会保证
      每个类别的样本都按同样比例分到三个集合里。

    - ZINC：官方直接给了 train/val/test 三个文件，直接用官方的划分。
"""

import json
import os
import sys

import torch
import torch.nn.functional as F

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from utils import DATA_DIR   # noqa: E402

# 本任务支持的数据集清单
TU_DATASETS = ["MUTAG", "PROTEINS", "ENZYMES", "IMDB-BINARY"]
ZINC_NAME = "ZINC"
ALL_DATASETS = TU_DATASETS + [ZINC_NAME]

# 任务类型：分类用 CrossEntropy+准确率，回归用 L1Loss+MAE
TASK_TYPE = {name: "classification" for name in TU_DATASETS}
TASK_TYPE[ZINC_NAME] = "regression"


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def compute_degree(edge_index, num_nodes):
    """统计每个节点的度数（有几条边连到它）。"""
    deg = torch.zeros(num_nodes, dtype=torch.long)
    deg.index_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.long))
    return deg


def degree_one_hot(edge_index, num_nodes, max_degree):
    """
    把节点度数变成 one-hot 向量作为节点特征。

    为什么要这么做？
        像 IMDB-BINARY 这种数据集只给了"谁和谁有连接"，没给节点特征。
        没有特征就没法跑 GNN，所以学术界通用的办法就是用度数当特征。
        例如度数=3 -> [0,0,0,1,0,0,...]（第 3 位是 1）
    """
    deg = compute_degree(edge_index, num_nodes).clamp(max=max_degree)
    return F.one_hot(deg, num_classes=max_degree + 1).float()


def stratified_split(data_list, labels, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    分层划分：保证每个类别都按 val_ratio / test_ratio 的比例分到验证集和测试集。

    做法：把样本按类别分组 -> 每一组内部各自打乱、各自切分 -> 再拼起来。
    """
    g = torch.Generator().manual_seed(seed)
    labels = torch.as_tensor(labels)

    train_idx, val_idx, test_idx = [], [], []
    for cls in torch.unique(labels):
        idx = (labels == cls).nonzero(as_tuple=False).view(-1)
        perm = idx[torch.randperm(idx.numel(), generator=g)]

        n = perm.numel()
        n_test = max(int(round(n * test_ratio)), 1)     # 至少留 1 个
        n_val = max(int(round(n * val_ratio)), 1)
        # 防止类别样本太少导致训练集为空
        if n - n_test - n_val < 1:
            n_test = max(n // 5, 1)
            n_val = max(n // 5, 1)

        test_idx.append(perm[:n_test])
        val_idx.append(perm[n_test:n_test + n_val])
        train_idx.append(perm[n_test + n_val:])

    train_idx = torch.cat(train_idx)
    val_idx = torch.cat(val_idx)
    test_idx = torch.cat(test_idx)

    # 最后再打乱一次，避免同类样本挤在一起影响 batch 统计（BatchNorm 尤其在意）
    train_idx = train_idx[torch.randperm(train_idx.numel(), generator=g)]
    val_idx = val_idx[torch.randperm(val_idx.numel(), generator=g)]
    test_idx = test_idx[torch.randperm(test_idx.numel(), generator=g)]

    return ([data_list[i] for i in train_idx],
            [data_list[i] for i in val_idx],
            [data_list[i] for i in test_idx])


# ---------------------------------------------------------------------------
# TUDataset 系列
# ---------------------------------------------------------------------------
def load_tu_dataset(name, val_ratio=0.1, test_ratio=0.1, seed=42, verbose=True):
    """
    加载一个 TUDataset 图分类数据集，并切分成训练/验证/测试。

    返回：
        train_list, val_list, test_list : 三个 Data 列表
        num_features : 节点特征维度
        num_classes  : 类别数
        task_type    : 'classification'
    """
    from torch_geometric.datasets import TUDataset

    # 【路径说明】TUDataset 自己把 raw 目录定义成 root/<数据集名>/raw，
    # 所以这里 root 直接给 data/，它就会去 data/MUTAG/raw/ 下面找文件，
    # 正好和 prepare_data.py 下载解压的位置对上。
    # 如果写成 root=data/TU，它就会去找 data/TU/MUTAG/raw/，那就找不到了。
    os.makedirs(DATA_DIR, exist_ok=True)

    # use_node_attr=True 表示"如果数据集自带节点特征就把它带上"
    dataset = TUDataset(root=DATA_DIR, name=name, use_node_attr=True)

    # 取出所有图。PyG 的切片方式 ds[i] 会返回第 i 张图
    data_list = [dataset[i] for i in range(len(dataset))]

    if verbose:
        print(f"  [{name}] 共 {len(data_list)} 张图，"
              f"节点特征维度={dataset.num_features}，类别数={dataset.num_classes}")

    # ---- 处理"没有节点特征"的数据集（比如 IMDB-BINARY）----
    if data_list[0].x is None:
        # 先统计整个数据集里的最大度数，用来决定 one-hot 多长
        max_deg = max(int(compute_degree(d.edge_index, d.num_nodes).max())
                      for d in data_list)
        max_deg = min(max_deg, 100)      # 上限设 100，避免特征维度太大
        if verbose:
            print(f"  [{name}] 该数据集没有节点特征，"
                  f"用节点度数 one-hot 代替（最大度数={max_deg}）")
        for d in data_list:
            d.x = degree_one_hot(d.edge_index, d.num_nodes, max_deg)

    # 统一类型：特征 float32、标签 int64
    for d in data_list:
        d.x = d.x.float()
        d.y = d.y.view(-1).long()

    labels = [int(d.y.item()) for d in data_list]
    train_list, val_list, test_list = stratified_split(
        data_list, labels, val_ratio, test_ratio, seed)

    if verbose:
        print(f"  [{name}] 划分后：训练 {len(train_list)} / "
              f"验证 {len(val_list)} / 测试 {len(test_list)}")

    return (train_list, val_list, test_list,
            data_list[0].x.size(1), dataset.num_classes, "classification")


# ---------------------------------------------------------------------------
# ZINC（分子图回归）
# ---------------------------------------------------------------------------
class ZINCDataset:
    """
    ZINC 数据集的自定义读取类。

    【为什么不用 PyG 自带的 ZINC？】
        PyG 自带的 ZINC 需要从 dropbox 下载 molecules.zip，国内网络拿不到。
        所以 prepare_data.py 改从 HuggingFace 镜像下载了同样内容的 jsonl 文件。
        这里就负责把这些 jsonl 读成 PyG 的 Data 对象。

    jsonl 每行长这样（一行 = 一张分子图）：
        {"node_feat": [[0],[1],...],        # 每个节点的原子类型
         "edge_index": [[0,1,...],[1,2,...]],  # 边的两端
         "edge_attr": [1,1,2,...],          # 化学键类型
         "y": [3.046],                      # 要预测的性质
         "num_nodes": 33}

    【缓存机制】
        读 22 万行 json 再转成 Data 对象挺慢的（几十秒）。
        所以第一次读完之后会存成一个 .pt 文件，下次就直接 load，快很多。
    """

    def __init__(self, max_graphs=10000, verbose=True):
        self.root = os.path.join(DATA_DIR, "ZINC")
        self.raw_dir = os.path.join(self.root, "raw")
        self.processed_dir = os.path.join(self.root, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)
        self.max_graphs = max_graphs
        self.verbose = verbose

        # 缓存文件名带上 max_graphs，避免用不同子集时读到错的缓存
        tag = "full" if not max_graphs else f"n{max_graphs}"
        self.cache = os.path.join(self.processed_dir, f"zinc_{tag}.pt")

    def _read_jsonl(self, path, limit):
        """把一个 jsonl 文件读成 Data 列表。"""
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                obj = json.loads(line)
                # 原子类型：one-hot 编码成向量
                atom = torch.tensor([a[0] for a in obj["node_feat"]], dtype=torch.long)
                x = F.one_hot(atom.clamp(min=0, max=self.num_atom_types - 1),
                              num_classes=self.num_atom_types).float()
                edge_index = torch.tensor(obj["edge_index"], dtype=torch.long)
                edge_attr = torch.tensor(obj["edge_attr"], dtype=torch.long)
                y = torch.tensor(obj["y"], dtype=torch.float).view(1)
                items.append((x, edge_index, edge_attr, y))
        return items

    def load(self):
        """
        返回 (train_list, val_list, test_list, num_features, 1, 'regression')
        """
        import torch as _t
        if os.path.exists(self.cache):
            if self.verbose:
                print(f"  [ZINC] 读取缓存 {os.path.basename(self.cache)}")
            obj = _t.load(self.cache)
            splits = obj["splits"]
            num_atom_types = obj["num_atom_types"]
        else:
            # 先扫一遍训练集，确定原子类型总数（决定 one-hot 的维度）
            self.num_atom_types = self._count_atom_types()
            splits = {}
            for split in ["train", "val", "test"]:
                path = os.path.join(self.raw_dir, f"{split}.jsonl")
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"找不到 {path}。请先在项目根目录运行：python prepare_data.py --task 3")
                limit = self.max_graphs if split == "train" else max(self.max_graphs // 10, 1000)
                if self.verbose:
                    print(f"  [ZINC] 读取 {split}.jsonl（最多 {limit} 张图）...")
                raw = self._read_jsonl(path, limit)
                splits[split] = raw
            num_atom_types = self.num_atom_types
            _t.save({"splits": splits, "num_atom_types": num_atom_types}, self.cache)
            if self.verbose:
                print(f"  [ZINC] 已缓存到 {os.path.basename(self.cache)}")

        from torch_geometric.data import Data

        def to_data_list(raw):
            out = []
            for x, edge_index, edge_attr, y in raw:
                out.append(Data(x=x, edge_index=edge_index,
                                edge_attr=edge_attr, y=y))
            return out

        train_list = to_data_list(splits["train"])
        val_list = to_data_list(splits["val"])
        test_list = to_data_list(splits["test"])

        if self.verbose:
            print(f"  [ZINC] 训练 {len(train_list)} / 验证 {len(val_list)} / "
                  f"测试 {len(test_list)} 张图，原子类型 one-hot 维度={num_atom_types}")

        return train_list, val_list, test_list, num_atom_types, 1, "regression"

    def _count_atom_types(self):
        """扫描训练集，统计一共有多少种原子类型。"""
        path = os.path.join(self.raw_dir, "train.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"找不到 {path}。请先在项目根目录运行：python prepare_data.py --task 3")
        mx = 0
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 20000:       # 扫 2 万行足够确定最大值了
                    break
                obj = json.loads(line)
                for a in obj["node_feat"]:
                    if a[0] > mx:
                        mx = a[0]
        return mx + 1


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------
def load_dataset(name, max_graphs=10000, seed=42, verbose=True):
    """
    统一的数据加载入口。返回：
        train_list, val_list, test_list, num_features, num_classes, task_type
    """
    if name in TU_DATASETS:
        return load_tu_dataset(name, seed=seed, verbose=verbose)
    elif name.upper() == ZINC_NAME:
        return ZINCDataset(max_graphs=max_graphs, verbose=verbose).load()
    else:
        raise ValueError(f"不支持的数据集：{name}。可选：{ALL_DATASETS}")


if __name__ == "__main__":
    print("=" * 78)
    print("检查图分类数据集")
    print("=" * 78)
    for name in TU_DATASETS:
        print(f"\n>>> {name}")
        try:
            tr, va, te, nf, nc, tt = load_dataset(name)
            print(f"    节点特征维度={nf}, 类别数={nc}, 任务类型={tt}")
            print(f"    示例图: x={tuple(tr[0].x.shape)}, "
                  f"edge_index={tuple(tr[0].edge_index.shape)}, y={tr[0].y.item()}")
            print("    [OK]")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"    [FAIL] {type(e).__name__}: {e}")

    print(f"\n>>> {ZINC_NAME}（只读 2000 张图做快速检查）")
    try:
        tr, va, te, nf, nc, tt = load_dataset(ZINC_NAME, max_graphs=2000)
        print(f"    节点特征维度={nf}, 任务类型={tt}")
        print(f"    示例图: x={tuple(tr[0].x.shape)}, "
              f"edge_index={tuple(tr[0].edge_index.shape)}, y={tr[0].y.item():.4f}")
        print("    [OK]")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"    [FAIL] {type(e).__name__}: {e}")
