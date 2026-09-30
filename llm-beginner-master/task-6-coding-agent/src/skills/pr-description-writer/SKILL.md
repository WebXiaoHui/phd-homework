---
name: pr-description-writer
description: 任务里出现"PR/描述/提交说明/change log/改动摘要"等词时加载。约定先 git_diff 看清改动，再写结构化描述。
---

# Skill: pr-description-writer

触发：issue/task 提到 PR、描述、改动说明、commit message。

## 流程

1. 先 `git_diff`（或 read 改动文件）搞清**到底改了什么**，不要凭猜。
2. 用下面的骨架写描述，中英皆可但前后一致：

```
## 背景
（这个 issue/缺陷是什么）
## 改动
- （文件/函数 + 一句改动说明，逐个列）
## 测试
- （跑了什么测试、结果）
```

3. 一段话不超过 ~200 字；重点写「修了什么 + 怎么验证」，不写过程流水账。

## 常见坑

- 没看 diff 就写 → 描述与真实改动不符。
- 只写"修复 bug"没有细节 → 信息量为零。
