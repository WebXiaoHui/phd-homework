"""calculator 工具：安全算术 + 数学函数。

用 AST 白名单求值（不做 `eval`），只放行：
- 常量 / Name(常数 pi,e,tau,inf) / 四则、整除、取模、幂、位运算、一元正负、比较；
- 调用白名单数学函数（sqrt/abs/round/pow/floor/ceil/log/exp/三角函数/…）。

float 结果同时给出完整精度与「6 位小数」两种写法，便于下游 agent 原样引用
（例如 sqrt(2026) 期望 45.011110）。
"""
import ast
import math
import operator as op

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "计算数学表达式（含开方等常用数学函数）。输入必须是单行 Python 数学表达式，"
            "例如 '(123 + 456) * 789'、'sqrt(2026)'、'2**10'。支持 + - * / // % ** 与括号、"
            "max/min/sum 以及 sqrt, abs, round, pow, floor, ceil, sin, cos, tan, "
            "log, log2, log10, exp, pi, e, gcd 等。不能包含赋值、import 或任意代码。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的数学表达式，例如 '(123+456)*789' 或 'sqrt(2026)'。",
                }
            },
            "required": ["expression"],
        },
    },
}

# 常数（Name 节点可直接引用）
_NAMES = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}

# 可调用的白名单（Call 节点）
_FUNCS = {
    "abs": abs, "round": round, "pow": pow,
    "max": max, "min": min, "sum": sum, "len": len,
    "gcd": math.gcd, "sqrt": math.sqrt, "cbrt": getattr(math, "cbrt", None),
    "floor": math.floor, "ceil": math.ceil, "trunc": math.trunc,
    "fabs": math.fabs, "factorial": math.factorial,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "atan2": math.atan2, "degrees": math.degrees, "radians": math.radians,
    "log": math.log, "log2": math.log2, "log10": math.log10, "exp": math.exp,
    "expm1": math.expm1, "hypot": math.hypot, "fmod": math.fmod,
    "isinf": math.isinf, "isnan": math.isnan, "copysign": math.copysign,
}
_FUNCS = {k: v for k, v in _FUNCS.items() if v is not None}

_BINOPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow,
    ast.BitAnd: op.and_, ast.BitOr: op.or_, ast.BitXor: op.xor,
    ast.LShift: op.lshift, ast.RShift: op.rshift,
}
_UNARY = {ast.UAdd: op.pos, ast.USub: op.neg, ast.Invert: op.invert}
_CMP = {
    ast.Eq: op.eq, ast.NotEq: op.ne, ast.Lt: op.lt, ast.LtE: op.le,
    ast.Gt: op.gt, ast.GtE: op.ge,
}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool, complex)) or node.value is None:
            return node.value
        raise ValueError(f"不支持的常量：{node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in _NAMES:
            return _NAMES[node.id]
        raise ValueError(f"不支持的名称：{node.id}（只能引用 pi/e/tau/inf）")
    if isinstance(node, ast.BinOp):
        fn = _BINOPS.get(type(node.op))
        if fn is None:
            raise ValueError(f"不支持的二元运算符：{type(node.op).__name__}")
        return fn(_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        fn = _UNARY.get(type(node.op))
        if fn is None:
            raise ValueError(f"不支持的一元运算符：{type(node.op).__name__}")
        return fn(_eval(node.operand))
    if isinstance(node, ast.Compare):
        left = _eval(node.left)
        for opn, comp in zip(node.ops, node.comparators):
            fn = _CMP.get(type(opn))
            if fn is None:
                raise ValueError(f"不支持的比较运算符：{type(opn).__name__}")
            if not fn(left, _eval(comp)):
                return False
            left = _eval(comp)
        return True
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ValueError("只允许调用白名单数学函数（如 sqrt/floor/log）")
        fn = _FUNCS[node.func.id]
        args = [_eval(a) for a in node.args]
        kwargs = {}
        for kw in node.keywords:
            if kw.arg is not None:
                kwargs[kw.arg] = _eval(kw.value)
            else:
                raise ValueError("不支持 ** 展开")
        return fn(*args, **kwargs)
    if isinstance(node, ast.IfExp):
        return _eval(node.body) if _eval(node.test) else _eval(node.orelse)
    raise ValueError(f"不支持的语法节点：{type(node).__name__}")


def _format_value(v):
    if v is None:
        return "None"
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, int):
        # 整数顺带给出位数，方便"结果几位数"这类问题让模型原样引用
        return f"{v}（{len(str(abs(v)))} 位整数）"
    if isinstance(v, complex):
        return str(v)
    if isinstance(v, float):
        if v != v:  # NaN
            return "nan"
        if v in (math.inf, -math.inf):
            return str(v)
        if v.is_integer() and abs(v) < 1e16:
            return str(int(v))
        full = repr(v)
        six = format(v, ".6f")
        if six in full:
            return full
        # 去掉 .6f 可能带的科学计数法尾巴，保持可读
        return f"{full}（6 位小数：{six}）"
    return str(v)


def run(args):
    if not isinstance(args, dict) or "expression" not in args:
        raise ValueError("calculator 需要参数 expression（str）")
    expr = args["expression"]
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("expression 必须是非空字符串")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
        value = _eval(tree.body)
    except SyntaxError as e:
        raise ValueError(f"表达式语法错误：{e.msg}（位置 {e.offset}）") from e
    except (ValueError, TypeError, ZeroDivisionError, ArithmeticError,
            OverflowError) as e:
        raise ValueError(f"计算失败：{e}") from e
    return _format_value(value)


if __name__ == "__main__":
    for e in ["2 + 3 * 4", "(123 + 456) * 789", "sqrt(2026)", "2 ** 10"]:
        print(f"{e} = {run({'expression': e})}")
