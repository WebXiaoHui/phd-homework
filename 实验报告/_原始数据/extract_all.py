# -*- coding: utf-8 -*-
"""按人工读出的 y 轴刻度值，把每张曲线图的评测指标"像素 -> 数值"地读出来。

要点：
  * 刻度值来自 `strips_*.png`（把各子图 y 轴刻度标签区裁剪拼接后人工读取），
    这里给出的是**完整标签序列（自上而下）**。
  * 白色网格线检测会漏掉贴着坐标轴上/下边缘（5% 以内）的刻度，所以不能简单地把
    "最上面一条网格线"当成"最上面那个标签"。本脚本按几何关系自动对齐：
        相邻网格线间距 Δf（占轴长比例）≈ 刻度步长 s / 轴范围 span
        最上面那条**被检测到**的网格线，其上方还应有 ktop = floor(f0/Δf) 个刻度
    再要求 ktop + kbot = 标签数 - 检测到的网格线数 做一致性校验，不一致就告警。
  * 单点子图（画不出线）：该点必在坐标轴正中 -> 取轴中心行插值。
  * 曲线子图：给出曲线的 min / max / 末端值（末端 = 最右列橙色像素）。
"""
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, r"D:\PHD\博一\入学任务\实验报告\_原始数据")
from solve_values import axes_boxes, gridlines

BASE = r"D:\PHD\博一\入学任务\transformers_tasks-main"

# (标题, 图相对路径, 子图号, [自上而下的刻度标签值])
SPEC = [
    ("文本分类 BERT", r"text_classification\logs\comment_classify\BERT.png", 1, [0.33, 0.32, 0.31]),
    ("文本分类 BERT", r"text_classification\logs\comment_classify\BERT.png", 2, [0.35, 0.34, 0.33]),
    ("文本分类 BERT", r"text_classification\logs\comment_classify\BERT.png", 3, [0.33, 0.32, 0.31]),
    ("文本分类 BERT", r"text_classification\logs\comment_classify\BERT.png", 4,
     [0.270, 0.265, 0.260, 0.255, 0.250]),

    ("文本匹配 PointWise", r"text_matching\supervised\logs\comment_classify\ERNIE-PointWise.png", 1,
     [0.98, 0.96, 0.94, 0.92, 0.90]),
    ("文本匹配 PointWise", r"text_matching\supervised\logs\comment_classify\ERNIE-PointWise.png", 2,
     [0.90, 0.88, 0.86, 0.84, 0.82]),
    ("文本匹配 PointWise", r"text_matching\supervised\logs\comment_classify\ERNIE-PointWise.png", 3,
     [0.96, 0.94, 0.92, 0.90, 0.88]),
    ("文本匹配 PointWise", r"text_matching\supervised\logs\comment_classify\ERNIE-PointWise.png", 4,
     [0.94, 0.92, 0.90, 0.88, 0.86]),

    ("文本匹配 DSSM", r"text_matching\supervised\logs\comment_classify\ERNIE-DSSM.png", 1,
     [0.7, 0.6, 0.5, 0.4, 0.3]),
    ("文本匹配 DSSM", r"text_matching\supervised\logs\comment_classify\ERNIE-DSSM.png", 2,
     [0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]),
    ("文本匹配 DSSM", r"text_matching\supervised\logs\comment_classify\ERNIE-DSSM.png", 3,
     [0.9, 0.8, 0.7, 0.6]),
    ("文本匹配 DSSM", r"text_matching\supervised\logs\comment_classify\ERNIE-DSSM.png", 4,
     [0.600, 0.575, 0.550, 0.525, 0.500, 0.475, 0.450]),

    ("文本匹配 SentenceBERT", r"text_matching\supervised\logs\comment_classify\Sentence-Ernie.png", 1,
     [0.9, 0.8, 0.7, 0.6, 0.5]),
    ("文本匹配 SentenceBERT", r"text_matching\supervised\logs\comment_classify\Sentence-Ernie.png", 2,
     [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]),
    ("文本匹配 SentenceBERT", r"text_matching\supervised\logs\comment_classify\Sentence-Ernie.png", 3,
     [0.8, 0.7, 0.6, 0.5, 0.4]),
    ("文本匹配 SentenceBERT", r"text_matching\supervised\logs\comment_classify\Sentence-Ernie.png", 4,
     [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]),

    ("无监督 SimCSE", r"text_matching\unsupervised\simcse\logs\LCQMC\ERNIE-ESimCSE.png", 1,
     [0.63, 0.62, 0.61, 0.60, 0.59, 0.58]),
    ("无监督 SimCSE", r"text_matching\unsupervised\simcse\logs\LCQMC\ERNIE-ESimCSE.png", 2,
     [0.59, 0.58, 0.57, 0.56, 0.55, 0.54]),
    ("无监督 SimCSE", r"text_matching\unsupervised\simcse\logs\LCQMC\ERNIE-ESimCSE.png", 3,
     [1.00, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88]),
    ("无监督 SimCSE", r"text_matching\unsupervised\simcse\logs\LCQMC\ERNIE-ESimCSE.png", 4,
     [0.707, 0.706, 0.705, 0.704, 0.703, 0.702]),
    ("无监督 SimCSE", r"text_matching\unsupervised\simcse\logs\LCQMC\ERNIE-ESimCSE.png", 5,
     [0.55, 0.50, 0.45, 0.40]),

    ("PET", r"prompt_tasks\PET\logs\comment_classify\BERT-PET.png", 1,
     [0.78, 0.76, 0.74, 0.72, 0.70, 0.68, 0.66]),
    ("PET", r"prompt_tasks\PET\logs\comment_classify\BERT-PET.png", 2, [0.80, 0.78, 0.76, 0.74]),
    ("PET", r"prompt_tasks\PET\logs\comment_classify\BERT-PET.png", 3,
     [0.78, 0.76, 0.74, 0.72, 0.70, 0.68, 0.66]),
    ("PET", r"prompt_tasks\PET\logs\comment_classify\BERT-PET.png", 4,
     [0.76, 0.74, 0.72, 0.70, 0.68, 0.66]),

    ("p-tuning", r"prompt_tasks\p-tuning\logs\comment_classify\BERT.png", 1,
     [0.65, 0.64, 0.63, 0.62, 0.61, 0.60]),
    ("p-tuning", r"prompt_tasks\p-tuning\logs\comment_classify\BERT.png", 2,
     [0.760, 0.758, 0.756, 0.754, 0.752, 0.750]),
    ("p-tuning", r"prompt_tasks\p-tuning\logs\comment_classify\BERT.png", 3,
     [0.65, 0.64, 0.63, 0.62, 0.61, 0.60]),
    ("p-tuning", r"prompt_tasks\p-tuning\logs\comment_classify\BERT.png", 4,
     [0.640, 0.635, 0.630, 0.625, 0.620]),

    ("RLHF 奖励模型", r"RLHF\logs\reward_model\sentiment_analysis\ERNIE Reward Model.png", 1,
     [0.66, 0.64, 0.62, 0.60, 0.58]),

    ("问答生成 T5", r"answer_generation\logs\DuReaderQG\T5-Base-Chinese.png", 1, [0.08, 0.06, 0.04]),
    ("问答生成 T5", r"answer_generation\logs\DuReaderQG\T5-Base-Chinese.png", 2,
     [0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.00]),
    ("问答生成 T5", r"answer_generation\logs\DuReaderQG\T5-Base-Chinese.png", 3, [0.03, 0.02, 0.01, 0.00]),
    ("问答生成 T5", r"answer_generation\logs\DuReaderQG\T5-Base-Chinese.png", 4,
     [0.020, 0.015, 0.010, 0.005, 0.000]),
]

PARAM = {
    "文本分类 BERT": {1: "eval/accuracy", 2: "eval/precision", 3: "eval/recall", 4: "eval/f1"},
    "文本匹配 PointWise": {1: "eval/accuracy", 2: "eval/precision", 3: "eval/recall", 4: "eval/f1"},
    "文本匹配 DSSM": {1: "eval/accuracy", 2: "eval/precision", 3: "eval/recall", 4: "eval/f1"},
    "文本匹配 SentenceBERT": {1: "eval/accuracy", 2: "eval/precision", 3: "eval/recall", 4: "eval/f1"},
    "无监督 SimCSE": {1: "eval/accuracy", 2: "eval/precision", 3: "eval/recall", 4: "eval/f1",
                  5: "eval/spearman_corr"},
    "PET": {1: "eval/accuracy", 2: "eval/precision", 3: "eval/recall", 4: "eval/f1"},
    "p-tuning": {1: "eval/accuracy", 2: "eval/precision", 3: "eval/recall", 4: "eval/f1"},
    "RLHF 奖励模型": {1: "eval/accuracy"},
    "问答生成 T5": {1: "eval/bleu-size-1", 2: "eval/bleu-size-2", 3: "eval/bleu-size-3",
                4: "eval/bleu-size-4"},
}


def grid_rows(gray, box):
    x0, y0, x1, y1 = box
    sub = gray[y0:y1 + 1, x0:x1 + 1]
    h = sub.shape[0]
    off = int(h * 0.05)
    frac = (sub[off:int(h * 0.95), :] > 250).mean(axis=1)
    idx = np.nonzero(frac > 0.55)[0]
    out = []
    if len(idx):
        s = p = idx[0]
        for k in idx[1:]:
            if k - p > 2:
                out.append(y0 + off + (s + p) / 2.0)
                s = k
            p = k
        out.append(y0 + off + (s + p) / 2.0)
    return out


def main():
    cache = {}
    for title, rel, idx, labels in SPEC:
        path = BASE + "\\" + rel
        if path not in cache:
            im = Image.open(path).convert('RGB')
            a = np.asarray(im).astype(np.int16)
            _, boxes = axes_boxes(a)
            cache[path] = (im, a, boxes)
        im, a, boxes = cache[path]
        box = boxes[idx]
        gray = np.asarray(im.convert('L'))
        _, hf, na = gridlines(gray, a, box)
        rows = grid_rows(gray, box)
        x0, y0, x1, y1 = box
        n = len(hf)
        df = float(np.median(np.diff(sorted(hf)))) if n > 1 else 0.0
        warn = ''
        ktop = 0
        if n > 1 and len(labels) > n:
            ktop_geo = int(np.floor(hf[0] / df + 0.08))
            kbot_geo = int(np.floor((1.0 - hf[-1]) / df + 0.08))
            need = len(labels) - n
            ok = False
            for cand in (ktop_geo, ktop_geo - 1, ktop_geo + 1):
                if cand < 0 or cand > need:
                    continue
                if need - cand <= kbot_geo + 1:
                    ktop, ok = cand, True
                    break
            if not ok:
                ktop = 0
                warn = '  ⚠配对存疑(标签数=%d, 网格线=%d, f0=%.3f, df=%.3f)' % (len(labels), n, hf[0], df)
        if ktop + n > len(labels):
            ktop = 0
            warn += '  ⚠回退'
        t_top = labels[ktop]
        t_bot = labels[ktop + n - 1]
        r_top, r_bot = rows[0], rows[-1]
        k = (t_bot - t_top) / (r_bot - r_top)

        def val(r):
            return t_top + (r - r_top) * k

        ys, xs = np.nonzero(na)
        name = PARAM[title][idx]
        if len(ys) == 0:
            center = (y0 + y1) / 2.0
            print('%-16s %-18s 单点 值=%.4f  (轴中心插值; 用到刻度 %g~%g, 标签%d个/网格%d条)%s'
                  % (title, name, val(center), t_top, t_bot, len(labels), n, warn))
        else:
            vmin, vmax = val(y0 + ys.max()), val(y0 + ys.min())
            last = ys[xs >= xs.max() - 2]
            print('%-16s %-18s 曲线 min=%.4f max=%.4f 末端=%.4f  (刻度 %g~%g; 标签%d/网格%d; px=%d)%s'
                  % (title, name, vmin, vmax, val(y0 + last.mean()), t_top, t_bot,
                     len(labels), n, len(ys), warn))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
