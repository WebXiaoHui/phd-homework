# -*- coding: utf-8 -*-
"""
datasets.py —— 知识图谱数据集的读取与划分（任务四：知识图谱）

【什么是知识图谱？】

    知识图谱就是把"事实"存成一堆三元组 (头实体, 关系, 尾实体)：

        (姚明, 出生于, 上海)
        (上海, 属于, 中国)
        (中国, 首都, 北京)

    用符号写就是 (h, r, t) —— head, relation, tail。
    上面的例子里，"姚明""上海""中国"是**实体**，"出生于""属于""首都"是**关系**。

【知识图谱补全（KGC）在做什么？】

    给一个不完整的三元组，让模型猜缺的那一块：

        链接预测（猜尾实体）：(姚明, 出生于, ?)   -> 应该填"上海"
        链接预测（猜头实体）：(?, 首都, 北京)     -> 应该填"中国"

    本质上和任务二的链路预测很像，但多了一个"关系"维度：
    任务二只有"有没有边"，这里要区分"是哪一种边"。
    而且实体数量动辄几万，没法像任务二那样直接算全图 GNN，
    所以主流做法是**给每个实体和关系学一个向量（embedding）**，
    再用一个打分函数判断三元组成不成立。

【数据集】

    WN18RR  : 从 WordNet 里抽出来的。40943 个实体，11 种关系。
              特点是关系很少但语义规整，模型容易学好。
    FB15k-237: 从 Freebase 里抽出来的。14541 个实体，237 种关系。
              关系多、更贴近真实场景，也更难。
              名字里的 237 就是把"会导致测试集泄漏的关系"删掉之后剩下的关系数。

【训练集 / 验证集 / 测试集 怎么划分？（这是任务四要求了解的知识点）】

    三个文件 train.txt / valid.txt / test.txt，每行一个三元组，用制表符分隔。

    划分的原则是：**同一种关系的三元组会同时出现在三个集合里**。
    也就是说，验证/测试集里会出现训练集里见过的"关系"，
    但"头实体-关系-尾实体"这个具体组合必须是训练集里没见过的。

    为什么不能像普通分类那样随机切？
        如果随机切，可能出现 (姚明,出生于,上海) 在训练集，
        (姚明,出生于,?) 在测试集 —— 那模型只要"背答案"就行了，
        根本没学到东西。所以切分要保证"具体的三元组不重复"。

    另外，实体表（entity2id）和关系表（relation2id）是从**三个集合合起来**
    建的，所以测试集里出现的实体训练时也见过（只是没见过这个组合）。

【过滤式评估（filtered evaluation）—— 一个非常重要的细节】

    评测的时候，我们要给正确答案在所有实体里排个名次。
    假设正确答案是 (姚明, 出生于, 上海)，模型给"上海"排第 3 名，
    看起来还不错？但如果 (姚明, 出生于, 上海) 之外，
    还有 (姚明, 出生于, 松江) 这种**同样正确**的答案呢？

    知识图谱里有大量这种情况（一个人可以有多个出生地记录）。
    如果不处理，这些"也对"的答案会挤在"上海"前面，名次被白白拉低。

    所以标准做法是：**排名前先把所有已知的正确答案从候选里删掉**
    （这就是 "filtered" 的意思）。本文件会预先建好这些过滤表。

    注意：过滤时用的是 train + valid + test 里**所有**出现过的真三元组，
    这是学术界公认的做法（叫 "full filtered setting"）。
"""

import os
import sys
from collections import defaultdict

import torch

# Windows 控制台默认 GBK 编码，改成 UTF-8 避免打印中文/符号时报错
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from utils import DATA_DIR   # noqa: E402

# 本任务支持的数据集
DATASET_NAMES = ["WN18RR", "FB15k-237"]


# ---------------------------------------------------------------------------
# 数据容器
# ---------------------------------------------------------------------------
class KGDataset:
    """
    保存知识图谱的所有内容。字段说明：

        num_entities : 实体总数 E
        num_relations: 关系总数 R
        train / valid / test : 三个集合的三元组，每个是 (3, N) 的 long 张量
                               第 0 行是头实体 id，第 1 行是关系 id，第 2 行是尾实体 id

        hr2tails : dict, (h, r) -> set(所有已知的尾实体)
        rt2heads : dict, (r, t) -> set(所有已知的头实体)

        hr2tails / rt2heads 就是为了"过滤式评估"准备的：
            评估"猜尾实体"时，用 hr2tails 把 (h, r) 对应的所有正确答案删掉；
            评估"猜头实体"时，用 rt2heads 把 (r, t) 对应的所有正确答案删掉。
    """

    def __init__(self, name, num_entities, num_relations,
                 train, valid, test, hr2tails, rt2heads):
        self.name = name
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.train = train
        self.valid = valid
        self.test = test
        self.hr2tails = hr2tails
        self.rt2heads = rt2heads

    def describe(self):
        return (f"{self.name}: 实体数={self.num_entities}, 关系数={self.num_relations}, "
                f"训练三元组={self.train.size(1)}, "
                f"验证三元组={self.valid.size(1)}, "
                f"测试三元组={self.test.size(1)}")

    def to(self, device):
        """把三个集合的张量搬到指定设备上（过滤表留在 CPU 上，因为要按 Python 字典查）。"""
        self.train = self.train.to(device)
        self.valid = self.valid.to(device)
        self.test = self.test.to(device)
        return self


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------
def _find_data_dir(name):
    """
    找到数据集所在目录。

    prepare_data.py 解压后可能会有两种结构（取决于压缩包内部长什么样）：
        data/WN18RR/train.txt               <- 直接解压
        data/WN18RR/WN18RR/train.txt        <- 压缩包里还套了一层同名目录
    这里两种都试一下，找到哪个用哪个，免得用户手动挪文件。
    """
    candidates = [
        os.path.join(DATA_DIR, name),
        os.path.join(DATA_DIR, name, name),
    ]
    for d in candidates:
        if os.path.exists(os.path.join(d, "train.txt")):
            return d
    raise FileNotFoundError(
        f"在 {candidates} 里都没找到 train.txt。\n"
        f"请先在项目根目录运行：python prepare_data.py --task 4")


def _read_triples(path, e2id, r2id):
    """
    读一个 .txt 文件，把每一行 "头实体\\t关系\\t尾实体" 变成 id 三元组。

    实体/关系表是"边读边建"的：遇到没见过的名字就分配一个新 id。
    所以要先读 train，再读 valid / test，这样 id 从 0 开始排。
    """
    heads, rels, tails = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                # 有些文件用空格分隔，兜底再切一次
                parts = line.split()
            if len(parts) != 3:
                continue
            h, r, t = parts
            if h not in e2id:
                e2id[h] = len(e2id)
            if t not in e2id:
                e2id[t] = len(e2id)
            if r not in r2id:
                r2id[r] = len(r2id)
            heads.append(e2id[h])
            rels.append(r2id[r])
            tails.append(e2id[t])

    return torch.tensor([heads, rels, tails], dtype=torch.long)


def load_dataset(name, verbose=True):
    """
    读取一个知识图谱数据集。

    返回一个 KGDataset 对象。
    """
    if name not in DATASET_NAMES:
        raise ValueError(f"不支持的数据集：{name}。可选：{DATASET_NAMES}")

    d = _find_data_dir(name)

    # 实体表和关系表从三个集合合起来建（这样测试集的实体在训练时也见过）
    e2id, r2id = {}, {}
    train = _read_triples(os.path.join(d, "train.txt"), e2id, r2id)
    valid = _read_triples(os.path.join(d, "valid.txt"), e2id, r2id)
    test = _read_triples(os.path.join(d, "test.txt"), e2id, r2id)

    # ---- 建过滤表 ----
    # 注意：这里把 train + valid + test 全部放进去，是学术界的标准做法
    # （叫 full filtered setting）。这样"任何已知的正确答案"都会被过滤掉。
    hr2tails = defaultdict(set)
    rt2heads = defaultdict(set)
    for triples in (train, valid, test):
        h_all, r_all, t_all = triples[0].tolist(), triples[1].tolist(), triples[2].tolist()
        for h, r, t in zip(h_all, r_all, t_all):
            hr2tails[(h, r)].add(t)
            rt2heads[(r, t)].add(h)

    ds = KGDataset(name, len(e2id), len(r2id), train, valid, test,
                   dict(hr2tails), dict(rt2heads))

    if verbose:
        print(f"  [{name}] 从 {d} 读取")
        print(f"  {ds.describe()}")
    return ds


# ---------------------------------------------------------------------------
# 负采样
# ---------------------------------------------------------------------------
def sample_negatives(triples, num_neg, num_entities, device, seed=None):
    """
    给一批正样本三元组抽负样本。

    【怎么造负样本？】—— "替换法"，这是知识图谱补全的通用做法：
        随机把**头实体**或**尾实体**换成另一个随机实体，关系保持不变。

        正样本：(姚明, 出生于, 上海)
        负样本：(姚明, 出生于, 北京)   <- 换了尾实体，这个事实是假的
        负样本：(周杰伦, 出生于, 上海) <- 换了头实体，这个事实也是假的

        注意：**关系不能换**。因为换关系会造出"关系类型不对"的样本，
        而不是"事实不对"的样本，学不到有用的东西。

    【为什么用 corrupter 方式而不是用过滤表筛？】
        理论上应该筛掉"其实也是真的"的负样本，但那样太慢
        （每个样本都要查一次字典）。学术界普遍的做法是**不筛**，
        直接随机替换，因为随机撞上真答案的概率非常低
        （1/实体数，WN18RR 是 1/40943），影响可以忽略。
        这一点在 RotatE 原论文里也是这样处理的。

    返回：
        (2*B*num_neg, 3) 的负样本张量（一半换头、一半换尾）
    """
    h, r, t = triples[0], triples[1], triples[2]
    B = h.size(0)

    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)

    # 每条正样本抽 num_neg 个负样本：一半换头，一半换尾
    n_head = num_neg // 2
    n_tail = num_neg - n_head

    negs = []
    if n_head > 0:
        rand_h = torch.randint(0, num_entities, (B, n_head),
                               device=device, generator=generator)
        negs.append(torch.stack([
            rand_h.reshape(-1),
            r.unsqueeze(1).expand(B, n_head).reshape(-1),
            t.unsqueeze(1).expand(B, n_head).reshape(-1),
        ], dim=0))
    if n_tail > 0:
        rand_t = torch.randint(0, num_entities, (B, n_tail),
                               device=device, generator=generator)
        negs.append(torch.stack([
            h.unsqueeze(1).expand(B, n_tail).reshape(-1),
            r.unsqueeze(1).expand(B, n_tail).reshape(-1),
            rand_t.reshape(-1),
        ], dim=0))

    return torch.cat(negs, dim=1)


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 78)
    print("检查知识图谱数据集")
    print("=" * 78)
    for name in DATASET_NAMES:
        print(f"\n>>> {name}")
        try:
            ds = load_dataset(name)

            # 检查1：三元组不能越界
            for split_name, tri in [("train", ds.train), ("valid", ds.valid),
                                    ("test", ds.test)]:
                assert tri[0].max() < ds.num_entities
                assert tri[2].max() < ds.num_entities
                assert tri[1].max() < ds.num_relations
            print("    范围检查：所有实体/关系 id 都在合法范围内 [OK]")

            # 检查2：三个集合之间不能有重复的三元组（否则就是信息泄漏）
            def key_set(tri):
                return set(zip(tri[0].tolist(), tri[1].tolist(), tri[2].tolist()))

            k_tr, k_va, k_te = key_set(ds.train), key_set(ds.valid), key_set(ds.test)
            print(f"    训练∩验证={len(k_tr & k_va)}  训练∩测试={len(k_tr & k_te)}  "
                  f"验证∩测试={len(k_va & k_te)}  （都应该是 0）")

            # 检查3：负采样能不能正常工作
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tri = ds.train[:, :8].to(dev)
            neg = sample_negatives(tri, 4, ds.num_entities, dev, seed=0)
            print(f"    负采样自测：8 条正样本 -> {neg.size(1)} 条负样本 "
                  f"（8 × 4 = 32）")
            same = (neg[0] == tri[0].repeat_interleave(4)) & \
                   (neg[2] == tri[2].repeat_interleave(4))
            print(f"    其中有 {int(same.sum())} 条两头都没换（应该是 0）")
            print("    [OK]")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"    [FAIL] {type(e).__name__}: {e}")
