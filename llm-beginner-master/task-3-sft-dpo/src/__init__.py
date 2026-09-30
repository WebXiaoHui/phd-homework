"""task-3-sft-dpo：指令微调与偏好对齐（SFT + DPO）。

组成：
- lora.py    手写 LoRA 注入 / 合并 / adapter 存取
- chat.py    Qwen chat template + loss masking
- dataset.py MOSS SFT 数据 / DPO 偏好数据解析与打包
- compare.py base / SFT / DPO 三模型同指令对比（报告素材）
"""
