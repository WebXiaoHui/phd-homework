# -*- coding: utf-8 -*-
"""把若干张曲线图的 **y 轴刻度标签区** 裁剪出来，拼成一张大图，
方便一次读图就拿到所有子图的刻度值。

用法（Windows python）:
  python build_strips.py <out.png> <chart1.png> [<chart2.png> ...]

输出图：
  每一行 = 一张曲线图；行内从左到右 = 该图第 1..5 个子图（跳过第 0 个 train_loss）
  的 y 轴刻度标签条，条上标注 `图名#子图号`。
"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, r"D:\PHD\博一\入学任务\实验报告\_原始数据")
from solve_values import axes_boxes

FONT = "C:/Windows/Fonts/arial.ttf"


def main():
    out = sys.argv[1]
    paths = sys.argv[2:]
    rows = []
    for p in paths:
        im = Image.open(p).convert('RGB')
        a = np.asarray(im).astype(np.int16)
        _, boxes = axes_boxes(a)
        name = p.replace('\\', '/').split('/')[-1].replace('.png', '')
        strips = []
        for idx, (x0, y0, x1, y1) in enumerate(boxes):
            if idx == 0:
                continue
            sx0 = max(0, x0 - 175)
            sy0 = max(0, y0 - 25)
            sx1 = min(im.width, x0 + 12)
            sy1 = min(im.height, y1 + 25)
            s = im.crop((sx0, sy0, sx1, sy1))
            canvas = Image.new('RGB', (s.width, s.height + 34), 'white')
            d = ImageDraw.Draw(canvas)
            try:
                f = ImageFont.truetype(FONT, 26)
            except Exception:
                f = ImageFont.load_default()
            d.text((4, 4), '%s#%d' % (name[:22], idx), fill='red', font=f)
            canvas.paste(s, (0, 34))
            strips.append(canvas)
        W = sum(s.width for s in strips) + 10 * len(strips)
        H = max(s.height for s in strips) + 10
        row = Image.new('RGB', (W, H), 'white')
        x = 5
        for s in strips:
            row.paste(s, (x, 5))
            x += s.width + 10
        rows.append(row)
    W = max(r.width for r in rows)
    H = sum(r.height for r in rows) + 12 * len(rows)
    big = Image.new('RGB', (W, H), (200, 200, 200))
    y = 6
    for r in rows:
        big.paste(r, (4, y))
        y += r.height + 12
    big.save(out)
    print('saved', out, big.size)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
