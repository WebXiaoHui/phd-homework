# -*- coding: utf-8 -*-
"""把训练曲线图里某条曲线"像素 -> 数值"地读出来。

用法（Windows python）：
  python curve_read.py <chart.png> <subplot_index> <top_tick_value> <bottom_tick_value>

其中 `subplot_index` 为 extract_metrics.py 输出的子图序号（0 起，按行优先 = 左上→右上→中左…），
`top_tick_value` / `bottom_tick_value` 是该子图最上面/最下面那条**白色网格线**对应的 y 轴刻度值
（人工从图上读出，例如 0.94 与 0.86）。

输出：
  * 该子图内橙色曲线（iTrainingLogger 用 color='darkorange' 画）的
    - 数值范围（min / max）
    - **最后一个点**的值（取最右侧一列橙色像素，即训练结束时的取值）
    - **最优值**（分类指标取 max、loss 取 min，两个数都打印，看哪个用得上）
  * 如果该子图没有橙色像素，则说明它只记录了 1 个点（画不出线），
    此时按 extract_metrics.py 的结论：该点的值 = 该子图 y 轴范围中点 = 最上/最下刻度值的平均。
"""
import sys
import numpy as np
from PIL import Image

ORANGE = np.array([255, 140, 0], dtype=np.int16)
BG = np.array([234, 234, 242], dtype=np.int16)
WHITE = 250


def subplot_boxes(path):
    im = Image.open(path).convert('RGB')
    a = np.asarray(im).astype(np.int16)
    bg = np.abs(a - BG).sum(axis=2) < 12
    H, W = bg.shape
    cell = 10
    gh, gw = (H + cell - 1) // cell, (W + cell - 1) // cell
    grid = np.zeros((gh, gw), dtype=bool)
    ys, xs = np.nonzero(bg)
    grid[ys // cell, xs // cell] = True
    seen = np.zeros_like(grid)
    out = []
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
                sub = bg[y0:y1, x0:x1]
                yy, xx = np.nonzero(sub)
                if sub.sum() >= 20000:
                    out.append((x0 + xx.min(), y0 + yy.min(), x0 + xx.max(), y0 + yy.max()))
    out.sort(key=lambda b: ((b[1] + 100) // 400, b[0]))
    return im, a, out


def gridline_rows(gray, box):
    x0, y0, x1, y1 = box
    sub = gray[y0:y1 + 1, x0:x1 + 1]
    h, w = sub.shape
    rows = sub[int(h * 0.08):int(h * 0.92), :] > WHITE
    frac = rows.mean(axis=1)
    idx = np.nonzero(frac > 0.55)[0]
    groups = []
    if len(idx):
        s = p = idx[0]
        for k in idx[1:]:
            if k - p > 2:
                groups.append((s + p) / 2.0)
                s = k
            p = k
        groups.append((s + p) / 2.0)
    return [y0 + int(h * 0.08) + g for g in groups]


def main():
    path, idx = sys.argv[1], int(sys.argv[2])
    t_top, t_bot = float(sys.argv[3]), float(sys.argv[4])
    im, a, boxes = subplot_boxes(path)
    box = boxes[idx]
    x0, y0, x1, y1 = box
    rows = gridline_rows(np.asarray(im.convert('L')), box)
    if not rows:
        print('!! 没有检测到网格线'); return
    r_top, r_bot = rows[0], rows[-1]
    scale = -(t_top - t_bot) / (r_bot - r_top)         # 行号往下增加，数值下降 -> 斜率为负
    def val(r):
        return t_top + (r - r_top) * scale
    print('subplot#%d box=(%d,%d)-(%d,%d)' % (idx, x0, y0, x1, y1))
    print('  网格线像素行: %s' % rows)
    print('  顶刻度 %.6g @row %d , 底刻度 %.6g @row %d , 每像素 %.3g' % (t_top, r_top, t_bot, r_bot, scale))
    sub = a[y0:y1 + 1, x0:x1 + 1]
    om = np.abs(sub - ORANGE).sum(axis=2) < 150
    ys, xs = np.nonzero(om)
    if len(ys) == 0:
        print('  该子图没有曲线（只有 1 个点，画不出线）')
        print('  ⇒ 单点值 ≈ (%.6g + %.6g)/2 = %.6g  （前提：该子图只记录过一次评测）'
              % (t_top, t_bot, (t_top + t_bot) / 2))
        return
    vmin, vmax = val(y0 + ys.max()), val(y0 + ys.min())
    xr = xs.max()
    last = ys[xs >= xr - 2]
    vlast = val(y0 + last.mean())
    print('  曲线像素数 %d ；x 像素范围 %d..%d' % (len(ys), xs.min(), xs.max()))
    print('  数值范围: min=%.4f  max=%.4f' % (vmin, vmax))
    print('  最后一个点(最右端, 约 %.0f%% 处) = %.4f' % (100.0 * xr / (x1 - x0), vlast))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
