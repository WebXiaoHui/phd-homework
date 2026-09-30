# -*- coding: utf-8 -*-
"""从 iTrainingLogger 生成的训练曲线图中"反解"出被记录的数据值（无需 OCR）。

原理（已实测验证）：
  * 画布 seaborn-darkgrid：坐标轴底色 #EAEAF2，网格线白色 -> 先框出子图。
    **注意**：坐标轴区域的上下边界要用「BG 像素的行计数」来找，不能用连通域 bbox
    （相邻子图之间会被细碎的 BG 像素连起来，导致 bbox 变高、刻度比例算错）。
  * iTrainingLogger.record() 用 ax.plot() 画线（无 marker）。若某子图只记录过**一个**点，
    线不可见，但坐标轴范围是 matplotlib 单点 autoscale 的结果：
        lim = (v - 5.5%|v|, v + 5.5%|v|)   →  span = 0.11·v
    相邻网格线间距占轴长比例 Δf = step / span = step / (0.11·v)
        ⇒ v = step / (0.11·Δf)
    对 step 取遍"漂亮数" {1,2,2.5,5}×10^k，再用 matplotlib 的 AutoLocator 校验
    刻度个数与相对位置，即可唯一定出 v（同时把 x 轴也解出来做交叉验证：
    单点 x 应当等于最后一次评测的 global_step）。

用法: python solve_values.py <chart.png> ...
"""
import sys
import numpy as np
from PIL import Image
from matplotlib.ticker import AutoLocator

ORANGE = np.array([255, 140, 0], dtype=np.int16)
BG = np.array([234, 234, 242], dtype=np.int16)
WHITE = 250


def axes_boxes(a):
    """先粗聚类找候选，再用 BG 行/列计数定出真实坐标轴范围。"""
    bg = np.abs(a - BG).sum(axis=2) < 12
    H, W = bg.shape
    cell = 10
    gh, gw = (H + cell - 1) // cell, (W + cell - 1) // cell
    grid = np.zeros((gh, gw), dtype=bool)
    ys, xs = np.nonzero(bg)
    grid[ys // cell, xs // cell] = True
    seen = np.zeros_like(grid)
    boxes = []
    for i in range(gh):
        for j in range(gw):
            if grid[i, j] and not seen[i, j]:
                st = [(i, j)]
                seen[i, j] = True
                cells = []
                while st:
                    ci, cj = st.pop()
                    cells.append((ci, cj))
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            ni, nj = ci + di, cj + dj
                            if 0 <= ni < gh and 0 <= nj < gw and grid[ni, nj] and not seen[ni, nj]:
                                seen[ni, nj] = True
                                st.append((ni, nj))
                y0 = min(c[0] for c in cells) * cell
                y1 = min(H, (max(c[0] for c in cells) + 1) * cell)
                x0 = min(c[1] for c in cells) * cell
                x1 = min(W, (max(c[1] for c in cells) + 1) * cell)
                if bg[y0:y1, x0:x1].sum() < 20000:
                    continue
                # 用列计数收紧左右边界
                colc = bg[y0:y1, :].sum(axis=0)
                cols = np.nonzero(colc > 0.25 * (y1 - y0))[0]
                cols = cols[(cols >= x0 - 20) & (cols <= x1 + 20)]
                if len(cols) == 0:
                    continue
                cx0, cx1 = int(cols.min()), int(cols.max())
                rowc = bg[:, cx0:cx1 + 1].sum(axis=1)
                rows = np.nonzero(rowc > 0.35 * (cx1 - cx0))[0]
                rows = rows[(rows >= y0 - 20) & (rows <= y1 + 20)]
                if len(rows) == 0:
                    continue
                # 取包含最多像素的那一段连续行
                segs, s = [], rows[0]
                for k in range(1, len(rows)):
                    if rows[k] - rows[k - 1] > 3:
                        segs.append((s, rows[k - 1]))
                        s = rows[k]
                segs.append((s, rows[-1]))
                # 取"起点最高、终点最低"的那段：曲线贴顶时中间会出现低 BG 行，
                # 不能只取最长的一段，否则坐标轴会被截短（PET 就踩过这个坑）。
                ry0 = int(min(t[0] for t in segs))
                ry1 = int(max(t[1] for t in segs))
                boxes.append((cx0, ry0, cx1, ry1))
    boxes = sorted(set(boxes), key=lambda b: ((b[1] + 60) // 400, b[0]))
    # --- 几何校正：同一张图里所有子图的高度应当一致；若某子图因曲线贴顶被截短，
    #     就把它吸附到"包含它、且高度正常"的公共行带（top/bottom 组合）上。
    if boxes:
        hmax = max(b[3] - b[1] for b in boxes)
        bands = sorted({(b[1], b[3]) for b in boxes if b[3] - b[1] >= 0.9 * hmax})
        fixed = []
        for (bx0, by0, bx1, by1) in boxes:
            if by1 - by0 >= 0.9 * hmax:
                fixed.append((bx0, by0, bx1, by1)); continue
            cand = [bd for bd in bands if bd[0] <= by0 and bd[1] >= by1 + 40]
            if cand:
                bd = min(cand, key=lambda t: t[1] - t[0])
                fixed.append((bx0, bd[0], bx1, bd[1]))
            else:
                fixed.append((bx0, by0, bx1, by1))
        boxes = sorted(set(fixed), key=lambda b: ((b[1] + 60) // 400, b[0]))
    return bg, boxes


def gridlines(gray, rgb, box):
    x0, y0, x1, y1 = box
    sub = gray[y0:y1 + 1, x0:x1 + 1]
    h, w = sub.shape
    rows = sub[int(h * 0.05):int(h * 0.95), :] > WHITE
    cols = sub[:, int(w * 0.05):int(w * 0.95)] > WHITE
    rfrac = rows.mean(axis=1)
    cfrac = cols.mean(axis=0)

    def peaks(frac, thresh=0.55):
        idx = np.nonzero(frac > thresh)[0]
        out = []
        if len(idx):
            s = p = idx[0]
            for k in idx[1:]:
                if k - p > 2:
                    out.append((s + p) / 2.0)
                    s = k
                p = k
            out.append((s + p) / 2.0)
        return out

    orow, ocol = int(h * 0.05), int(w * 0.05)
    hf = [(orow + p) / (h - 1) for p in peaks(rfrac)]     # 0=顶
    vf = [(ocol + p) / (w - 1) for p in peaks(cfrac)]     # 0=左
    na = np.abs(rgb[y0:y1 + 1, x0:x1 + 1] - ORANGE).sum(axis=2) < 150
    return vf, hf, na


def solve(fracs, cand_s, from_top=False):
    """在"漂亮步长 s" × "候选值 v" 的二维网格上搜索，使预测刻度位置与实测网格线吻合。

    关系式：span = s / Δf ；lim = (v-5.5%v, v+5.5%v)  ⇒  v = s / (0.11·Δf)（初值）
    再以初值为中心细搜，可把 v 精确到刻度所在的整数值（例如 0.32 而不是 0.31898）。
    """
    fracs = sorted(fracs)
    if len(fracs) < 2:
        return None
    df = float(np.median(np.diff(fracs)))
    best = None
    for st in cand_s:
        v0 = st / (0.11 * df)
        if not np.isfinite(v0) or v0 <= 0:
            continue
        for v in np.linspace(v0 * 0.9, v0 * 1.1, 2001):
            span = 0.055 * v
            lo, hi = v - span, v + span
            kmin = int(np.ceil(lo / st - 1e-9))
            kmax = int(np.floor(hi / st + 1e-9))
            if kmax - kmin + 1 != len(fracs):
                continue
            ticks = [k * st for k in range(kmin, kmax + 1)]
            if from_top:
                pred = [(hi - t) / (hi - lo) for t in reversed(ticks)]
            else:
                pred = [(t - lo) / (hi - lo) for t in ticks]
            err = float(np.max(np.abs(np.array(sorted(pred)) - np.array(fracs))))
            if best is None or err < best[0]:
                best = (err, float(v), st, ticks)
    return best


def main():
    cand = [b * 10 ** k for k in range(-7, 7) for b in (1, 2, 2.5, 5)]
    for path in sys.argv[1:]:
        im = Image.open(path).convert('RGB')
        a = np.asarray(im).astype(np.int16)
        gray = np.asarray(im.convert('L'))
        print('=' * 92)
        print('FILE:', path.replace('\\', '/').split('/')[-1])
        bg, boxes = axes_boxes(a)
        for i, box in enumerate(boxes):
            vf, hf, na = gridlines(gray, a, box)
            npx = int(na.sum())
            x0, y0, x1, y1 = box
            head = '  #%d box=(%d,%d)-(%d,%d) h=%d w=%d %s' % (
                i, x0, y0, x1, y1, y1 - y0, x1 - x0, 'CURVE' if npx >= 200 else 'single-point')
            print(head + ' | nH=%d H=%s | nV=%d V=%s' % (
                len(hf), ','.join('%.3f' % f for f in hf),
                len(vf), ','.join('%.3f' % f for f in vf)))
            if npx < 200:
                ry = solve(hf, cand, from_top=True)
                rx = solve(vf, cand)
                ys = ('y=%.6f (step %g, err %.4f)' % (ry[1], ry[2], ry[0])) if ry else 'y=?'
                xs = ('x=%.1f (step %g, err %.4f)' % (rx[1], rx[2], rx[0])) if rx else 'x=?'
                print('        ' + ys + '  ' + xs)
            else:
                print('        (curve px=%d)' % npx)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
