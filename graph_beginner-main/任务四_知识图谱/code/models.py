# -*- coding: utf-8 -*-
"""
models.py —— 三种知识图谱补全模型（任务四：知识图谱）

【三个模型的核心思想（先看这一句话版）】

    TransE  ：「关系 = 从头实体到尾实体的平移向量」
               h + r ≈ t
               例：(姚明) + (出生于) ≈ (上海)
               把"上海"和"姚明+出生于"算出来的向量放得越近，分数越高。
               简单、快，但处理不了"一对多""多对多"的关系
               （因为 (姚明,出生于,上海) 和 (姚明,出生于,松江) 要求
                上海和松江的向量都等于姚明+出生于，那它俩就分不开了）。

    RotatE  ：「关系 = 在复数空间里的旋转」
               t = h ∘ r，其中 r 的模长固定为 1（所以 r 只是个"旋转角度"）
               例：(姚明) 旋转一个角度得到 (上海)，旋转另一个角度得到 (松江)
               因为旋转是可逆的、能区分方向，所以能处理 TransE 处理不了的
               对称/反对称/一对多关系。

    ConvE   ：「用卷积神经网络来打分」
               把 h 和 r 拼成一个"二维图片"，用卷积核提取局部特征，
               再和 t 做内积。
               好处是引入了非线性，表达能力比纯线性模型强。

【三个模型共用的接口】（这样上层训练脚本可以一视同仁地调用）

    model.score(h_idx, r_idx, t_idx)    -> (B,) 每个三元组的分数，越大越可能是真的
    model.score_all_tails(h_idx, r_idx) -> (B, E) 给一批 (h, r) 打出"所有实体当尾实体"的分数
    model.score_all_heads(r_idx, t_idx) -> (B, E) 给一批 (r, t) 打出"所有实体当头实体"的分数

    后两个是**评估**用的：算 MRR / Hits@K 时要给所有候选实体排名。

【为什么所有模型都用同一套分数含义？】
    "分数越大越好"这个约定很重要。如果 TransE 用距离（越小越好）、
    RotatE 用相似度（越大越好），那训练和评估的代码就得写两套，
    很容易出 bug。所以本代码统一成"**分数越大 = 越可能是真的**"，
    TransE 的负距离会加上一个负号变成正分数。
"""

import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Windows 控制台默认是 GBK 编码，改成 UTF-8 避免中文/符号打印时报错
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _uniform_init(emb, scale=1.0):
    """
    均匀初始化 embedding。

    范围是 [-scale, scale]。这个做法来自 KGE 的经典实现，
    目的是让初始向量的长度和维度匹配（维度越大，初始化范围越小），
    这样模型一开始的分数不会因为维度不同而差好几个数量级。
    """
    nn.init.uniform_(emb.weight.data, -scale, scale)


# ===========================================================================
# 1. TransE —— Translating Embeddings (Bordes et al., 2013)
# ===========================================================================
class TransE(nn.Module):
    """
    TransE 的打分函数：

        距离 d(h, r, t) = || h + r - t ||     （L1 或 L2 范数）
        分数 score      = -d(h, r, t)          （取负号，这样越大越好）

    【为什么初始化时要把关系向量归一化？】
        因为关系向量起的是"平移"的作用。如果它一开始就很长，
        那 h + r 会跑到实体向量分布之外，训练很难收敛。
        所以把关系向量初始化为单位长度，是个常用的技巧。
    """

    def __init__(self, num_entities, num_relations, dim=500, p_norm=2):
        super().__init__()
        self.name = "TransE"
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.p_norm = p_norm

        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)

        # 初始化范围：和论文一致，用 6/sqrt(dim)
        bound = 6.0 / math.sqrt(dim)
        _uniform_init(self.entity_emb, bound)
        _uniform_init(self.relation_emb, bound)

        # 把关系向量归一化到单位长度（放在 no_grad 里，不参与梯度）
        with torch.no_grad():
            self.relation_emb.weight.data = F.normalize(
                self.relation_emb.weight.data, p=2, dim=1)

    # ---- 训练时用：给一批具体三元组打分 ----
    def score(self, h, r, t):
        hh = self.entity_emb(h)          # (B, d)
        rr = self.relation_emb(r)        # (B, d)
        tt = self.entity_emb(t)          # (B, d)
        # 负距离
        return -torch.norm(hh + rr - tt, p=self.p_norm, dim=-1)

    # ---- 评估时用：给一批 (h, r) 打出所有实体的分数 ----
    def score_all_tails(self, h, r):
        """
        【为什么不用 for 循环一个个算？】
            因为要算 B 条三元组 × E 个实体的分数，直接循环会非常慢。
            可以用展开公式把它变成一次矩阵乘法：

                ||q - t||² = ||q||² - 2·q·t + ||t||²

            其中 q = h + r。中间那项 q·t 正好是一个矩阵乘法
            (B, d) × (d, E) = (B, E)，GPU 算这个飞快。
        """
        q = self.entity_emb(h) + self.relation_emb(r)     # (B, d)
        T = self.entity_emb.weight                        # (E, d)
        q2 = (q * q).sum(dim=-1, keepdim=True)            # (B, 1)
        t2 = (T * T).sum(dim=-1).unsqueeze(0)             # (1, E)
        # clamp(min=0) 是为了防止浮点误差让结果变成很小的负数，开根号会得到 NaN
        d2 = (q2 - 2.0 * q @ T.t() + t2).clamp(min=0.0)   # (B, E)
        return -d2.sqrt()

    def score_all_heads(self, r, t):
        """猜头实体：q = t - r，然后和所有实体比距离。"""
        q = self.entity_emb(t) - self.relation_emb(r)
        H = self.entity_emb.weight
        q2 = (q * q).sum(dim=-1, keepdim=True)
        h2 = (H * H).sum(dim=-1).unsqueeze(0)
        d2 = (q2 - 2.0 * q @ H.t() + h2).clamp(min=0.0)
        return -d2.sqrt()


# ===========================================================================
# 2. RotatE —— Rotation in Complex Space (Sun et al., 2019)
# ===========================================================================
class RotatE(nn.Module):
    """
    RotatE 把实体和关系放到**复数空间**里。

    【复数怎么用代码表示？】
        PyTorch 里没有原生的复数 embedding 层（老版本），
        所以用一个长度 2*dim 的实数向量，前一半当实部、后一半当虚部。

        实体向量：长度 2*dim，切成 (实部, 虚部)
        关系向量：长度 dim  —— 注意只有一半！
                  它存的是"旋转角度 θ"，然后取 cos/sin 得到旋转因子：
                      r = cos(θ) + i·sin(θ)
                  这样 |r| = cos²+sin² = 1 恒成立，天然满足"模长为 1"的约束，
                  不需要额外加正则项。

    【打分函数】
        h ∘ r 的复数乘法展开：
            (a+bi)(c+di) = (ac - bd) + (ad + bc)i
        然后算它和 t 的距离：
            score = γ - ||h∘r - t||

        γ 是个常数（叫 margin），让分数有个基准。

    【为什么 RotatE 比 TransE 强？】
        因为"旋转"能区分方向，也能让 h∘r 同时靠近多个不同的 t
        （只要它们是圆周上的不同点），所以一对多关系也处理得了。
    """

    def __init__(self, num_entities, num_relations, dim=500, gamma=6.0):
        super().__init__()
        self.name = "RotatE"
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.emb_dim = dim            # 复数向量的维度
        self.gamma = gamma

        # 实体：实部 + 虚部，所以是 2*dim
        self.entity_emb = nn.Embedding(num_entities, dim * 2)
        # 关系：存的是角度，只有 dim 维
        self.relation_emb = nn.Embedding(num_relations, dim)

        # 初始化范围。RotatE 原论文用的就是这个范围
        self.embedding_range = (gamma + 2.0) / dim
        _uniform_init(self.entity_emb, self.embedding_range)
        # 关系角度初始化为 [-π, π] 之间的均匀分布
        pid = math.pi / self.embedding_range
        nn.init.uniform_(self.relation_emb.weight.data, -pid, pid)

    def _split(self, x):
        """把一个 (..., 2*dim) 的向量切成实部和虚部。"""
        return x[..., :self.emb_dim], x[..., self.emb_dim:]

    def _relation_phase(self, r):
        """把存的角度变成 cos/sin（也就是单位圆上的旋转因子）。"""
        phase = self.relation_emb(r) / (self.embedding_range / math.pi)
        return torch.cos(phase), torch.sin(phase)

    def score(self, h, r, t):
        re_h, im_h = self._split(self.entity_emb(h))
        re_t, im_t = self._split(self.entity_emb(t))
        re_r, im_r = self._relation_phase(r)

        # 复数乘法 h ∘ r
        re_score = re_h * re_r - im_h * im_r
        im_score = re_h * im_r + im_h * re_r

        # 和 t 的距离
        re_score = re_score - re_t
        im_score = im_score - im_t
        # stack 后 norm(dim=0) 算的是每个维度上的 sqrt(re² + im²)
        dist = torch.stack([re_score, im_score], dim=0).norm(dim=0).sum(dim=-1)
        return self.gamma - dist

    def _all_scores(self, q_re, q_im, E_re, E_im):
        """
        公共部分：算出 (B, E) 的分数矩阵。
        和 TransE 一样用展开公式把"距离"变成矩阵乘法。
        """
        q_re2 = (q_re * q_re).sum(dim=-1, keepdim=True)
        q_im2 = (q_im * q_im).sum(dim=-1, keepdim=True)
        e_re2 = (E_re * E_re).sum(dim=-1).unsqueeze(0)
        e_im2 = (E_im * E_im).sum(dim=-1).unsqueeze(0)

        d_re = (q_re2 - 2.0 * q_re @ E_re.t() + e_re2).clamp(min=0.0).sqrt()
        d_im = (q_im2 - 2.0 * q_im @ E_im.t() + e_im2).clamp(min=0.0).sqrt()
        return self.gamma - (d_re + d_im)

    def score_all_tails(self, h, r):
        re_h, im_h = self._split(self.entity_emb(h))
        re_r, im_r = self._relation_phase(r)
        # 旋转后的头实体 = h ∘ r
        q_re = re_h * re_r - im_h * im_r
        q_im = re_h * im_r + im_h * re_r
        E_re, E_im = self._split(self.entity_emb.weight)
        return self._all_scores(q_re, q_im, E_re, E_im)

    def score_all_heads(self, r, t):
        """
        猜头实体要反过来：已知 t 和 r，求 h 使得 h ∘ r ≈ t。
        两边同时"逆旋转"：h ≈ t ∘ r⁻¹，而 r⁻¹ = cos(θ) - i·sin(θ)（共轭）。
        """
        re_t, im_t = self._split(self.entity_emb(t))
        re_r, im_r = self._relation_phase(r)
        # 乘共轭：t ∘ conj(r)
        q_re = re_t * re_r + im_t * im_r
        q_im = -re_t * im_r + im_t * re_r
        E_re, E_im = self._split(self.entity_emb.weight)
        return self._all_scores(q_re, q_im, E_re, E_im)


# ===========================================================================
# 3. ConvE —— Convolutional 2D Knowledge Graph Embeddings (Dettmers et al., 2018)
# ===========================================================================
class ConvE(nn.Module):
    """
    ConvE 用一个 2D 卷积来给三元组打分。

    【为什么要"卷积"？】
        TransE 和 RotatE 的打分函数都是线性的（加减法、内积）。
        ConvE 的想法是：把实体和关系拼起来排成一张"二维小图"，
        用卷积核去提取 h 和 r 之间的**局部交互特征**，再经过几层全连接，
        这样就有了非线性，表达能力更强。

    【前向过程（看图理解）】

        1) 取出 h 的向量 (dim) 和 r 的向量 (dim)
        2) 拼起来 -> 长度 2*dim
        3) 重新排成二维：(emb_dim1, 2*dim/emb_dim1)
           例如 dim=200 时，400 个数排成 10 行 40 列
        4) 当成"单通道图片"送进 Conv2d(1, 32, kernel_size=3)
           -> 输出 32 张特征图
        5) 拉平 -> 全连接到 dim 维 -> 得到最终的 (h, r) 表示
        6) 和所有实体的向量做内积，得到 (B, 实体数) 的分数

    【注意】ConvE 的输入是"实体 id"，输出是所有实体上的分数，
    一次前向就能给所有实体打分，这叫 "1-N scoring"，
    是 ConvE 论文强调的效率优势。
    """

    def __init__(self, num_entities, num_relations, dim=200,
                 emb_dim1=10, out_channels=32, kernel_size=3,
                 input_dropout=0.2, hidden_dropout=0.3, feature_map_dropout=0.2):
        super().__init__()
        self.name = "ConvE"
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.emb_dim1 = emb_dim1
        self.kernel_size = kernel_size

        total = 2 * dim                     # 拼接后的长度
        # 下面这个断言是必须的：要把 total 个数整齐地排成 emb_dim1 行的二维矩阵，
        # 而且卷积核是 3×3、不加 padding，所以行和列都不能少于 3。
        assert total % emb_dim1 == 0, \
            f"2*dim={total} 必须能被 emb_dim1={emb_dim1} 整除，请换一个 dim"
        self.emb_dim2 = total // emb_dim1
        assert self.emb_dim2 >= kernel_size, \
            f"排成 {self.emb_dim1}×{self.emb_dim2} 之后列数太少，" \
            f"放不下 {kernel_size}×{kernel_size} 的卷积核，请把 dim 调大一点"

        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_emb = nn.Embedding(num_relations, dim)
        bound = 6.0 / math.sqrt(dim)
        _uniform_init(self.entity_emb, bound)
        _uniform_init(self.relation_emb, bound)

        # 卷积层：输入通道 1（因为只有一张"图"），输出 out_channels 张特征图
        self.conv2d = nn.Conv2d(1, out_channels,
                                (kernel_size, kernel_size), 1, 0,
                                bias=True)
        self.bn_conv = nn.BatchNorm2d(out_channels)

        # 卷积后的空间尺寸（那个 -kernel_size+1 就是"不加 padding 的卷积会缩小多少"）
        conved_h = self.emb_dim1 - kernel_size + 1
        conved_w = self.emb_dim2 - kernel_size + 1
        flat_size = out_channels * conved_h * conved_w

        # 全连接：把特征图压回 dim 维
        self.fc = nn.Linear(flat_size, dim)
        self.bn_fc = nn.BatchNorm1d(dim)
        # 每个实体还有一个偏置项（原论文的做法）
        self.bias = nn.Parameter(torch.zeros(num_entities))

        self.input_dropout = nn.Dropout(input_dropout)
        self.hidden_dropout = nn.Dropout(hidden_dropout)
        self.feature_map_dropout = nn.Dropout2d(feature_map_dropout)

    def _encode(self, h, r):
        """把 (h, r) 编码成 dim 维的向量。"""
        e1 = self.entity_emb(h)                       # (B, dim)
        e2 = self.relation_emb(r)                     # (B, dim)
        x = self.input_dropout(torch.cat([e1, e2], dim=-1))   # (B, 2*dim)

        # 排成二维"图片"： (B, 1, emb_dim1, emb_dim2)
        x = x.view(-1, 1, self.emb_dim1, self.emb_dim2)

        x = self.conv2d(x)                            # (B, C, H', W')
        x = self.bn_conv(x)
        x = F.relu(x)
        x = self.feature_map_dropout(x)

        x = x.view(x.size(0), -1)                     # 拉平
        x = self.fc(x)                                # (B, dim)
        x = self.hidden_dropout(x)
        x = self.bn_fc(x)
        x = F.relu(x)
        return x

    def score(self, h, r, t):
        """给具体的三元组打分：编码出来的向量和尾实体向量做内积。"""
        x = self._encode(h, r)                        # (B, dim)
        tt = self.entity_emb(t)                       # (B, dim)
        return (x * tt).sum(dim=-1) + self.bias[t]

    def score_all_tails(self, h, r):
        """一次给所有实体打分（ConvE 论文强调的 1-N scoring）。"""
        x = self._encode(h, r)                        # (B, dim)
        return x @ self.entity_emb.weight.t() + self.bias.unsqueeze(0)

    def score_all_heads(self, r, t, max_rows=32768):
        """
        猜头实体。

        注意：ConvE 的编码器是"h 和 r 一起进卷积"的，
        不能像 TransE/RotatE 那样把距离展开成一次矩阵乘法，
        所以只能老实枚举：把每个候选实体都当成 h，和 r 一起算一遍分数。

        【为什么要分块，而且分成"二维块"？】
            要算的分数总数 = 三元组数 B × 实体数 E，非常大
            （FB15k-237 的测试集是 20466 × 14541 ≈ 3 亿个）。
            但要**一次性编码**的行数只有 C（候选数）× B（三元组数），
            这个可以自由控制。

            所以下面开两个循环：
                - 外层按三元组分块（Bsub）
                - 内层按候选实体分块（C）
            并约束 C × Bsub <= max_rows，这样显存占用就不会随
            数据规模增长，既不会爆显存，又只需要一次前向。

            代价是这部分的计算量本身就是 E × B 次编码，逃不掉，
            所以 ConvE 的"猜头实体"评估比 TransE/RotatE 慢得多。
            这是 ConvE 这个模型的固有代价，不是实现问题。

        返回 (B, E) 的分数矩阵。
        """
        E = self.num_entities
        B = r.size(0)
        out = torch.empty(B, E, device=r.device, dtype=self.entity_emb.weight.dtype)
        tt = self.entity_emb(t)                       # (B, dim)

        C_chunk = min(E, 256)                         # 一次处理多少个候选实体
        B_sub = max(1, max_rows // C_chunk)           # 一次处理多少条三元组

        for b0 in range(0, B, B_sub):
            b1 = min(b0 + B_sub, B)
            rb = r[b0:b1]
            tb = tt[b0:b1]
            nb = b1 - b0
            pieces = []
            for c0 in range(0, E, C_chunk):
                c1 = min(c0 + C_chunk, E)
                nc = c1 - c0
                cand = torch.arange(c0, c1, device=r.device)
                # 每个候选实体都要和这一小批里每个 r 组合
                h_rep = cand.repeat_interleave(nb)     # (C*Bsub,)
                r_rep = rb.repeat(nc)                  # (C*Bsub,)
                t_rep = tb.repeat(nc, 1)               # (C*Bsub, dim)
                x = self._encode(h_rep, r_rep)         # (C*Bsub, dim)
                sc = (x * t_rep).sum(dim=-1) + self.bias[h_rep]
                pieces.append(sc.view(nc, nb).t())     # 变成 (Bsub, C)
            out[b0:b1] = torch.cat(pieces, dim=1)
        return out                                   # (B, E)


# ===========================================================================
# 工厂函数
# ===========================================================================
MODEL_NAMES = ["TransE", "RotatE", "ConvE"]

# 每个模型推荐的默认参数（想跑出接近论文的效果，用这些值）
DEFAULT_DIM = {"TransE": 500, "RotatE": 500, "ConvE": 200}
DEFAULT_MARGIN = {"TransE": 1.0, "RotatE": 6.0, "ConvE": 1.0}


def build_model(name, num_entities, num_relations, dim=None, margin=None, **kw):
    """
    根据名字创建模型。

    用法：
        model = build_model('RotatE', num_entities=40943, num_relations=11, dim=500)
    """
    if dim is None:
        dim = DEFAULT_DIM[name]
    if margin is None:
        margin = DEFAULT_MARGIN[name]

    key = name.lower()
    if key == "transe":
        return TransE(num_entities, num_relations, dim=dim)
    elif key == "rotate":
        return RotatE(num_entities, num_relations, dim=dim, gamma=margin)
    elif key == "conve":
        return ConvE(num_entities, num_relations, dim=dim)
    else:
        raise ValueError(f"未知模型：{name}。可选：{MODEL_NAMES}")


if __name__ == "__main__":
    print("=" * 78)
    print("模型自测（造一个小规模的知识图谱，看看三种模型跑不跑得通）")
    print("=" * 78)
    torch.manual_seed(0)

    E, R = 50, 5
    B = 7
    h = torch.randint(0, E, (B,))
    r = torch.randint(0, R, (B,))
    t = torch.randint(0, E, (B,))

    for name in MODEL_NAMES:
        for dim in ([64, 128] if name != "ConvE" else [40, 100]):
            model = build_model(name, E, R, dim=dim)
            s = model.score(h, r, t)
            st = model.score_all_tails(h, r)
            sh = model.score_all_heads(r, t)

            ok = (tuple(s.shape) == (B,)
                  and tuple(st.shape) == (B, E)
                  and tuple(sh.shape) == (B, E)
                  and torch.isfinite(s).all()
                  and torch.isfinite(st).all()
                  and torch.isfinite(sh).all())
            n_param = sum(p.numel() for p in model.parameters())
            status = "[OK]" if ok else "[FAIL]"
            print(f"  {name:7s} dim={dim:4d}  "
                  f"score={tuple(s.shape)}  all_tails={tuple(st.shape)}  "
                  f"all_heads={tuple(sh.shape)}  参数量={n_param:8,d}  {status}")

    # 额外验证一个几何直觉：RotatE 的关系向量模长必须恒为 1
    print("\n几何直觉检查：")
    m = build_model("RotatE", E, R, dim=32)
    re_r, im_r = m._relation_phase(torch.arange(R))
    modulus = (re_r ** 2 + im_r ** 2).sqrt()
    print(f"  RotatE 关系向量的模长 = {modulus.detach().numpy().round(6)}")
    print(f"  （全都是 1.0 才对，因为 cos²θ + sin²θ = 1）"
          f"  {'[OK]' if torch.allclose(modulus, torch.ones_like(modulus), atol=1e-5) else '[FAIL]'}")

    # 验证 TransE 的"平移"直觉：如果人为把 h+r 设成 t 的向量，分数应该最高
    print("\nTransE 平移直觉检查：")
    m = build_model("TransE", E, R, dim=8)
    with torch.no_grad():
        m.entity_emb.weight[0] = torch.tensor([1., 0, 0, 0, 0, 0, 0, 0])
        m.relation_emb.weight[0] = torch.tensor([0., 1, 0, 0, 0, 0, 0, 0])
        m.entity_emb.weight[1] = torch.tensor([1., 1, 0, 0, 0, 0, 0, 0])
    sc = m.score(torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))
    print(f"  令 h=[1,0,...], r=[0,1,...], t=[1,1,...]（正好 h+r=t）")
    print(f"  算出来的分数 = {float(sc):.6f}（因为 h+r-t 是零向量，距离为 0，"
          f"取负号就是 0）  {'[OK]' if abs(float(sc)) < 1e-5 else '[FAIL]'}")
