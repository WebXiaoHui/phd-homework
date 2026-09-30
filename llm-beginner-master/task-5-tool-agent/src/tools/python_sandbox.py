"""python_sandbox 工具：受限 Python 执行。

- import 一律禁止（子串黑名单）；
- builtins 白名单（print/sum/range/…），不给 eval/exec/open/getattr/globals 等逃逸口；
- 10 秒超时（signal 定时器；非主线程时自动退化为无超时）；
- stdout 捕获后返回给 agent；若代码没打印任何内容，自动回显末行表达式结果。

⚠️ 教学级防护：黑名单 + 白名单 + 超时只是"防手滑/防普通错误"，
挡不住 __class__.__mro__ / __globals__ / 内存耗尽等逃逸路径。只对可信、自产代码用；
对真正不可信输入请改用子进程 + resource / RestrictedPython / 容器隔离。
"""
import ast
import builtins as _builtins
import contextlib
import io
import re
import signal

# 与任务直接相关的黑名单 token（教学级，非安全边界）
_FORBIDDEN_RE = re.compile(
    r"(?i)(\bimport\b|__import__|open\s*\(|eval\s*\(|exec\s*\(|compile\s*\(|"
    r"__class__|__globals__|__subclasses__|__mro__|__builtins__|getattr\s*\(|"
    r"setattr\s*\(|globals\s*\(|locals\s*\(|vars\s*\(|help\s*\(|"
    r"os\s*\.|sys\s*\.|subprocess|socket|requests|urllib|shutil|pathlib|"
    r"base64|pickle|marshal)"
)

_ALLOWED_BUILTIN_NAMES = [
    # I/O 与基础
    "print", "len", "range", "sum", "min", "max", "abs", "round", "pow",
    "int", "float", "str", "bool", "complex", "bytes", "bytearray", "repr",
    "format", "ord", "chr", "hex", "oct", "bin", "hash", "divmod", "slice",
    "type", "isinstance", "issubclass", "super", "staticmethod", "classmethod",
    "property", "id",
    # 容器/迭代
    "list", "dict", "tuple", "set", "frozenset", "sorted", "reversed",
    "enumerate", "zip", "map", "filter", "all", "any", "iter", "next",
    # 异常
    "Exception", "ValueError", "TypeError", "RuntimeError", "KeyError",
    "IndexError", "StopIteration", "ArithmeticError", "ZeroDivisionError",
    "NotImplementedError", "NameError",
]
_ALLOWED_BUILTINS = {}
for _n in _ALLOWED_BUILTIN_NAMES:
    if hasattr(_builtins, _n):
        _ALLOWED_BUILTINS[_n] = getattr(_builtins, _n)
_ALLOWED_BUILTINS["True"] = True
_ALLOWED_BUILTINS["False"] = False
_ALLOWED_BUILTINS["None"] = None

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "python_sandbox",
        "description": (
            "在受限的 Python 环境里运行一段代码（禁止 import，安全白名单）。"
            "适合数值计算、算法练习、数据处理。代码请用 print() 输出最终结果；"
            "如果想看某个表达式的值但没写 print，工具会自动把最后一行表达式结果回显。"
            "不要 import 任何模块。例如：print(sum(range(100)))"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "要执行的 Python 代码（不能含 import）。",
                }
            },
            "required": ["code"],
        },
    },
}

TIMEOUT_SECONDS = 10


def _last_expr_value(tree, ns):
    """尝试求值最后一个『表达式语句』的值（仅用于无 print 时的自动回显）。"""
    if not getattr(tree, "body", None):
        return None
    last = tree.body[-1]
    if not isinstance(last, ast.Expr):
        return None
    allowed = (ast.Constant, ast.Name, ast.BinOp, ast.UnaryOp, ast.Call,
               ast.Compare, ast.IfExp, ast.List, ast.Tuple, ast.Dict,
               ast.Set, ast.Subscript)
    if not isinstance(last.value, allowed):
        return None
    try:
        code = compile(ast.Expression(last.value), "<sandbox-last>", "eval")
        return eval(code, ns)  # noqa: S307 —— ns 已受限
    except Exception:
        return None


def run(args):
    if not isinstance(args, dict) or "code" not in args:
        raise ValueError("python_sandbox 需要参数 code（str）")
    code = args["code"]
    if not isinstance(code, str) or not code.strip():
        raise ValueError("code 必须是非空字符串")

    if _FORBIDDEN_RE.search(code):
        m = _FORBIDDEN_RE.search(code)
        raise ValueError(f"代码包含被禁止的语句（{m.group(0)!r}）："
                         f"sandbox 不允许 import/open/eval/exec 及路径逃逸")

    # 1) 静态语法检查
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"代码语法错误：{e.msg}（行 {e.lineno}）") from e
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("sandbox 不允许 import 任何模块")

    # 2) 受限 globals 中 exec + stdout 捕获 + 超时
    safe_globals = {"__builtins__": _ALLOWED_BUILTINS}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            _exec_with_timeout(code, safe_globals)
    except Exception as e:  # 用户代码抛出的任何异常都反馈给 agent 纠错
        raise ValueError(f"代码执行出错：{type(e).__name__}: {e}") from e

    out = buf.getvalue()
    if not out:
        last_val = _last_expr_value(tree, safe_globals)
        if last_val is not None:
            out = repr(last_val)
    if not out:
        return "（代码没有输出。请用 print() 打印你想要的结果。）"
    return out.rstrip()


def _exec_with_timeout(code, ns):
    """带超时的 exec；仅在主线程可用 signal，否则退化为直接执行。"""
    import time

    def _alarm(*_):
        raise TimeoutError(f"执行超时（>{TIMEOUT_SECONDS} 秒）")

    old_handler = None
    armed = False
    try:
        old_handler = signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, TIMEOUT_SECONDS)
        armed = True
    except (ValueError, OSError, AttributeError):
        pass  # 非主线程：无法用信号，跳过超时（教学级，可接受）

    start = time.time()
    try:
        exec(compile(code, "<sandbox>", "exec"), ns)
    finally:
        if armed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            try:
                signal.signal(signal.SIGALRM, old_handler)
            except (ValueError, OSError):
                pass


if __name__ == "__main__":
    for c in ["print(sum(range(10)))",
              "def p(n):\n    return n > 1 and all(n % i for i in range(2, int(n**0.5)+1))\nprint(sum(i for i in range(100) if p(i)))"]:
        print("--- code ---")
        print(c)
        print("--- output ---")
        print(run({"code": c}))
