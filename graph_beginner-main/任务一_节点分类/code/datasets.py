# -*- coding: utf-8 -*-
"""
datasets.py —— 数据集加载（任务一：节点分类）

本任务用到三个数据集：
    Cora      ：2708 篇论文、5429 条引用关系、1433 维词袋特征、7 个类别
    Citeseer  ：3327 篇论文、4732 条引用关系、3703 维词袋特征、6 个类别
    Flickr    ：89250 张图片、899756 条共现边、500 维特征、7 个类别

【关于"训练集/验证集/测试集的划分"——本作业要求了解的知识点】
    Cora / Citeseer 用的是 Planetoid 论文提出的 **public split**：
        训练集：每个类别 20 个节点  -> 7*20 = 140 个
        验证集：500 个节点
        测试集：1000 个节点
        剩下的节点不参与训练也不参与评估（但是它们的特征和连边依然会被用到，
        也就是"半监督 / transductive"学习：测试节点在训练时是"看得见"的）。

    Flickr 用的是 GraphSAINT 论文给出的划分，比例大约是
        训练 50% / 验证 25% / 测试 25%。

    划分信息都保存在 data.train_mask / data.val_mask / data.test_mask 里，
    它们是和节点数等长的布尔向量，True 表示该节点属于这个集合。

【数据从哪来？】
    数据已经被 prepare_data.py 提前下载到 ../data/ 目录下了，
    所以这里的 Dataset 类不会再去联网。
    如果你是从零开始，请先在项目根目录执行：python prepare_data.py
"""

import os

from torch_geometric.datasets import Flickr, Planetoid

from utils import DATA_DIR

# 支持的数据集名字 -> 中文说明，方便打印
DATASET_INFO = {
    "Cora": "引文网络（2708 节点 / 10556 边 / 1433 特征 / 7 类）",
    "Citeseer": "引文网络（3327 节点 / 9104 边 / 3703 特征 / 6 类）",
    "Flickr": "图片共现网络（89250 节点 / 899756 边 / 500 特征 / 7 类）",
}


def load_dataset(name: str):
    """
    加载一个节点分类数据集。

    参数：
        name : 'Cora' / 'Citeseer' / 'Flickr'

    返回：
        data        : PyG 的 Data 对象，包含 x / edge_index / y / train_mask 等
        num_features: 每个节点的特征维度
        num_classes : 类别数

    用法：
        data, in_dim, n_cls = load_dataset('Cora')
    """
    if name in ("Cora", "Citeseer"):
        # Planetoid 会自动去 DATA_DIR/Cora/raw/ 找原始文件；
        # 因为我们已经提前下好了，所以它不会联网。
        dataset = Planetoid(root=DATA_DIR, name=name)
    elif name == "Flickr":
        # 【重要】PyG 的 Planetoid 会把 root 再拼上数据集名字（root/Cora/raw），
        # 但 Flickr 这个类没有做这件事，它的 raw 目录就是 root/raw。
        # 所以这里必须把 root 指到 DATA_DIR/Flickr，
        # 这样 raw 目录才正好是 DATA_DIR/Flickr/raw，和 prepare_data.py 下载的位置对上。
        dataset = Flickr(root=os.path.join(DATA_DIR, "Flickr"))
    else:
        raise ValueError(f"不支持的数据集：{name}。可选：Cora / Citeseer / Flickr")

    data = dataset[0]

    # 统一清理一下，保证后续代码不用关心细节：
    # 1) 特征转成 float32 —— Flickr 原始特征是 float64，直接用会让显存翻倍
    data.x = data.x.float()

    # 2) 标签转成 int64，PyTorch 的 CrossEntropyLoss 要求标签是 long 类型
    data.y = data.y.long()

    # 3) 检查三个 mask 是否都存在且非空。缺了的话后面训练就没法做了，早点报错更好。
    for mask_name in ("train_mask", "val_mask", "test_mask"):
        mask = getattr(data, mask_name, None)
        if mask is None or int(mask.sum()) == 0:
            raise RuntimeError(
                f"{name} 数据集的 {mask_name} 为空，数据可能没有正确下载。"
                f"请先在项目根目录执行：python prepare_data.py --task 1"
            )

    return data, dataset.num_features, dataset.num_classes


def dataset_summary(data) -> str:
    """生成一行数据集信息，方便写进日志。"""
    n_train = int(data.train_mask.sum())
    n_val = int(data.val_mask.sum())
    n_test = int(data.test_mask.sum())
    return (
        f"节点数={data.num_nodes}, 边数={data.num_edges}, "
        f"特征维度={data.num_features}, 类别数={int(data.y.max()) + 1}, "
        f"训练/验证/测试 = {n_train}/{n_val}/{n_test}"
    )


if __name__ == "__main__":
    # 直接运行本文件时，把三个数据集都加载一遍，检查数据是否正常
    print("=" * 78)
    print("检查数据集是否可以正常加载")
    print("=" * 78)
    for name in ["Cora", "Citeseer", "Flickr"]:
        print(f"\n>>> {name}  ({DATASET_INFO[name]})")
        try:
            data, in_dim, n_cls = load_dataset(name)
            print("    " + dataset_summary(data))
            print(f"    x.shape={tuple(data.x.shape)}, edge_index.shape={tuple(data.edge_index.shape)}")
            print("    ✅ 加载成功")
        except Exception as e:
            print(f"    ❌ 加载失败：{type(e).__name__}: {e}")
            print("    提示：请先到项目根目录运行  python prepare_data.py")
