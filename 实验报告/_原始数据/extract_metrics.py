# -*- coding: utf-8 -*-
"""从 iTrainingLogger 生成的训练曲线图中"反解"出被记录的数据值。

关键观察：
  * 画布 seaborn-darkgrid：坐标轴底色 #EAEAF2，网格线白色 -> 可先框出 6 个子图。
  * record() 用 ax.plot() 画线（无 marker）。若某子图**只记录过一个点**，线不可见，
    但坐标轴范围仍是 autoscale 结果：lim = (v-5%|v|, v+5%|v|)  （单点 -> nonsingular(expander=0.05)）
    于是"该点的值"= 轴范围中点；轴范围可由 白色网格线位置 + tick 间隔 反推：

        相邻网格线间距 Δf（占轴长比例）  =  tick步长 s / (0.1 * v)
        =>  v = s / (0.1 * Δf)

    遍历 s = {1,2,2.5,5}×10^k，用 matplotlib 的 AutoLocator 验证刻度个数与位置，即可唯一定出 v。

用法: python extract_metrics.py <chart.png> ...
"""
import sys
import numpy as np
from PIL import Image
from matplotlib.ticker import AutoLocator

ORANGE = np.array([255, 140, 0], dtype=np.int16)
BG = np.array([234, 234, 242], dtype=np.int16)
WHITE = 250


def bbox_clusters(mask, cell=10, min_px=20000):
    H, W = mask.shape
    gh, gw = (H + cell - 1) // cell, (W + cell - 1) // cell
    grid = np.zeros((gh, gw), dtype=bool)
    ys, xs = np.nonzero(mask)
    grid[ys // cell, xs // cell] = True
    seen = np.zeros_like(grid)
    out = []
    for i in range(gh):
        for j in range(gw):
            if grid[i, j] and not seen[i, j]:
                stack = [(i, j)]
                seen[i, j] = True
                cells = []
                while stack:
                    ci, cj = stack.pop()
                    cells.append((ci, cj))
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            ni, nj = ci + di, cj + dj
                            if 0 <= ni < gh and 0 <= nj < gw and grid[ni, nj] and not seen[ni, nj]:
                                seen[ni, nj] = True
                                stack.append((ni, nj))
                y0 = min(c[0] for c in cells) * cell
                y1 = min(H, (max(c[0] for c in cells) + 1) * cell)
                x0 = min(c[1] for c in cells) * cell
                x1 = min(W, (max(c[1] for c in cells) + 1) * cell)
                sub = mask[y0:y1, x0:x1]
                yy, xx = np.nonzero(sub)
                bb = (x0 + xx.min(), y0 + yy.min(), x0 + xx.max(), y0 + yy.max())
                n = int(sub.sum())
                if n >= min_px:
                    out.append((bb, n))
    return out


def gridlines(gray, box):
    x0, y0, x1, y1 = box
    sub = gray[y0:y1 + 1, x0:x1 + 1]
    h, w = sub.shape
    rows = sub[int(h * 0.08):int(h * 0.92), :] > WHITE
    cols = sub[:, int(w * 0.08):int(w * 0.92)] > WHITE
    row_frac = rows.mean(axis=1)
    col_frac = cols.mean(axis=0)

    def peaks(frac, thresh=0.55):
        idx = np.nonzero(frac > thresh)[0]
        groups = []
        if len(idx):
            start = prev = idx[0]
            for k in idx[1:]:
                if k - prev > 2:
                    groups.append((start + prev) / 2.0)
                    start = k
                prev = k
            groups.append((start + prev) / 2.0)
        return groups

    off_r = int(h * 0.08)
    off_c = int(w * 0.08)
    hf = [(off_r + p) / (h - 1) for p in peaks(row_frac)]     # 0=顶部
    vf = [(off_c + p) / (w - 1) for p in peaks(col_frac)]     # 0=左侧
    return vf, hf


def solve(fracs, cand_s):
    """fracs: 网格线相对位置(升序)。返回 [(err, v), ...] 按误差升序。"""
    if len(fracs) < 2:
        return []
    d = np.diff(np.array(fracs))
    df = float(np.median(d))
    loc = AutoLocator()
    res = []
    for s in cand_s:
        v = s / (0.1 * df)
        if not np.isfinite(v) or v <= 0:
            continue
        span = 0.05 * v
        lo, hi = v - span, v + span
        ticks = [t for t in loc.tick_values(lo, hi) if lo - 1e-9 <= t <= hi + 1e-9]
        if len(ticks) != len(fracs):
            continue
        # 位置：从轴的一端算起。fracs 升序；tick 值升序 -> 从"值大端"开始（顶部）
        pred = [(hi - t) / (hi - lo) for t in sorted(ticks, reverse=True)]
        err = float(np.max(np.abs(np.array(pred) - np.array(sorted(fracs)))))
        res.append((err, v, s))
    res.sort()
    return res


def main():
    cand_x = [b * 10 ** k for k in range(-2, 7) for b in (1, 2, 2.5, 5)]
    cand_y = [b * 10 ** k for k in range(-7, 6) for b in (1, 2, 2.5, 5)]
    for path in sys.argv[1:]:
        im = Image.open(path).convert('RGB')
        a = np.asarray(im).astype(np.int16)
        gray = np.asarray(im.convert('L'))
        print('=' * 88)
        print('FILE:', path.replace('\\', '/').split('/')[-1])
        bgmask = np.abs(a - BG).sum(axis=2) < 12
        omask = np.abs(a - ORANGE).sum(axis=2) < 150
        boxes = bbox_clusters(bgmask)
        boxes.sort(key=lambda t: ((t[0][1] + 100) // 400, t[0][0]))
        for i, (bb, n) in enumerate(boxes):
            x0, y0, x1, y1 = bb
            opx = int(omask[y0:y1 + 1, x0:x1 + 1].sum())
            vf, hf = gridlines(gray, bb)
            tag = 'CURVE(%d px)' % opx if opx >= 200 else 'single-point'
            rx = solve(vf, cand_x)
            ry = solve(hf, cand_y)
            print('  #%d box=(%d,%d)-(%d,%d) %s' % (i, x0, y0, x1, y1, tag))
            if rx:
                print('      x(user-x)=%.4f  (s=%g, err=%.4f)   next: %s'
                      % (rx[0][1], rx[0][2], rx[0][0], ['%.3f' % r[1] for r in rx[1:4]]))
            if ry:
                print('      y(value) =%.6f (s=%g, err=%.4f)   next: %s'
                      % (ry[0][1], ry[0][2], ry[0][0], ['%.4f' % r[1] for r in ry[1:4]]))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
