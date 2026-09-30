import numpy as np
from PIL import Image
import sys
sys.path.insert(0, r"D:\PHD\博一\入学任务\实验报告\_原始数据")
from solve_values import axes_boxes, gridlines
BASE = r"D:\PHD\博一\入学任务\transformers_tasks-main"
CHARTS = [
 ("文本分类 BERT", r"text_classification\logs\comment_classify\BERT.png"),
 ("文本匹配 PointWise", r"text_matching\supervised\logs\comment_classify\ERNIE-PointWise.png"),
 ("文本匹配 DSSM", r"text_matching\supervised\logs\comment_classify\ERNIE-DSSM.png"),
 ("文本匹配 SentenceBERT", r"text_matching\supervised\logs\comment_classify\Sentence-Ernie.png"),
 ("SimCSE", r"text_matching\unsupervised\simcse\logs\LCQMC\ERNIE-ESimCSE.png"),
 ("PET", r"prompt_tasks\PET\logs\comment_classify\BERT-PET.png"),
 ("p-tuning", r"prompt_tasks\p-tuning\logs\comment_classify\BERT.png"),
 ("RLHF", r"RLHF\logs\reward_model\sentiment_analysis\ERNIE Reward Model.png"),
 ("T5", r"answer_generation\logs\DuReaderQG\T5-Base-Chinese.png"),
]
for title, rel in CHARTS:
    im = Image.open(BASE + "\\" + rel).convert('RGB')
    a = np.asarray(im).astype(np.int16)
    _, boxes = axes_boxes(a)
    for idx, box in enumerate(boxes):
        if idx == 0: continue
        x0,y0,x1,y1 = box
        _,_,na = gridlines(np.asarray(im.convert('L')), a, box)
        ys,xs = np.nonzero(na)
        if len(ys) < 200: continue
        flags = []
        if ys.min() <= 2: flags.append('TOP-CLIP')
        if ys.max() >= (y1-y0) - 2: flags.append('BOTTOM-CLIP')
        if xs.min() <= 2: flags.append('LEFT')
        if xs.max() >= (x1-x0) - 2: flags.append('RIGHT')
        print('%-22s #%d  curve px=%5d  topoff=%3d botoff=%3d  %s' % (
            title, idx, len(ys), ys.min(), (y1-y0)-ys.max(), ','.join(flags) or 'ok'))
