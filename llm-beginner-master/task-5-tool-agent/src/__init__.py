"""task-5-tool-agent：手写 ReAct 循环 + 4 类本地工具（calculator / python_sandbox / file_search / wiki）。

组成：
- tools/calculator.py      安全算术/数学函数计算器（AST 白名单）
- tools/python_sandbox.py  受限 Python exec（import 黑名单 + builtins 白名单 + 超时）
- tools/file_search.py     本地目录文件名/内容检索（路径越界保护 + 内容片段预览）
- tools/wiki.py            维基百科 MediaWiki API 查询（中英文）
- agent.py                 ReActAgent：Thought/Action/Action Input/Observation 循环

注意：python_sandbox 的黑/白名单 + 超时只是教学级防护，不是真正的隔离，
只对可信 / 自产代码用。
"""
