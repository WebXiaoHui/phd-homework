# -*- coding: utf-8 -*-
"""分析 transformers_tasks-main 里 iTrainingLogger 生成的训练曲线图。

用法（Windows python）:
  python analyze_charts.py <chart.png> [<chart.png> ...]

输出：
  1) 每个子图 (subplot) 的坐标轴区域 bbox（通过 seaborn-darkgrid 背景色 #EAEAF2 检测）
  2) 每个子图里 darkorange(#FF8C00) 数据像素的聚类簇 bbox / 像素数
     —— 用来判断该子图记录了几个数据点、点的位置在哪里
"""
import sys
import numpy as np
from PIL import Image

ORANGE = np.array([255, 140, 0], dtype=np.int16)
BG = np.array([234, 234, 242], dtype=np.int16)   # seaborn-darkgrid axes facecolor


def clusters(mask, cell=20):
    """把 mask 里为 True 的像素按粗网格聚类（BFS），返回 [(bbox, npix), ...]"""
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
                out.append(((x0 + xx.min(), y0 + yy.min(), x0 + xx.max(), y0 + yy.max()), int(sub.sum())))
    out.sort(key=lambda t: (t[0][1], t[0][0]))
    return out


def axes_boxes(bgmask, min_area=20000):
    return [c for c in clusters(bgmask, cell=10) if c[1] > min_area]


def main():
    for path in sys.argv[1:]:
        im = Image.open(path).convert('RGB')
        a = np.asarray(im).astype(np.int16)
        print('=' * 100)
        print('FILE:', path.split('\\')[-1].split('/')[-1], 'size=', im.size)
        omask = np.abs(a - ORANGE).sum(axis=2) < 120
        bmask = np.abs(a - BG).sum(axis=2) < 12
        print('orange pixels:', int(omask.sum()), ' axes-bg pixels:', int(bmask.sum()))
        boxes = sorted(axes_boxes(bmask), key=lambda t: (t[0][1] // 200, t[0][0]))
        print('--- axes boxes (%d) ---' % len(boxes))
        for (bb, n) in boxes:
            print('   axes bbox x0=%d y0=%d x1=%d y1=%d  w=%d h=%d  bgpx=%d' %
                  (bb[0], bb[1], bb[2], bb[3], bb[2] - bb[0], bb[3] - bb[1], n))
        print('--- orange clusters ---')
        for (bb, n) in clusters(omask, cell=20):
            print('   cluster x0=%d y0=%d x1=%d y1=%d  w=%d h=%d  px=%d' %
                  (bb[0], bb[1], bb[2], bb[3], bb[2] - bb[0], bb[3] - bb[1], n))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
