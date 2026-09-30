import json
path = r"D:\PHD\博一\入学任务\llm-beginner-master\task-3-sft-dpo\data\dpo-en-zh-20k\dpo_zh.jsonl"
with open(path, "r", encoding="utf-8") as f:
    obj = json.loads(f.readline())
print("顶层字段：", list(obj.keys()))
print(json.dumps(obj, ensure_ascii=False)[:600])