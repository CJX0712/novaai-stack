"""NovaAI Stack — Agent 工具集（agent）。

作者：晨星
职责：为 ReAct 编排提供确定性工具（calculator / date / search）。
calculator 使用 AST 白名单安全求值，归一化全角标点为半角，且不剥离运算符字符；
date 返回人话时间；search 是检索工具的薄封装（注入 retriever 可独立验证）。
"""

from __future__ import annotations

import ast
import datetime
import re
from typing import Callable, List, Optional

from ..types import RetrievedChunk

# AST 白名单：仅允许四则运算与括号，杜绝代码注入
_ALLOWED = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Num,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
)

# 全角→半角运算符/数字映射（注意：归一化集合绝不能包含运算符字符）
_FW_MAP = {
    "×": "*", "÷": "/", "－": "-", "＋": "+", "（": "(", "）": ")",
    "，": ",", "。": "", "？": "", "！": "", "：": ":",
}
for _i in range(10):
    _FW_MAP[chr(0xFF10 + _i)] = str(_i)  # 全角数字 ０-９ -> 0-9

# 归一化时允许保留的运算符字符（绝不被 trim 掉）
_OP_CHARS = set("+-*/()%.0123456789")


def normalize_expr(text: str) -> str:
    out = []
    for ch in text:
        out.append(_FW_MAP.get(ch, ch))
    s = "".join(out)
    # 仅剥离尾部的非运算符标点/空白
    s = s.strip()
    while s and s[-1] not in _OP_CHARS:
        s = s[:-1].strip()
    return s


def calc(expr: str) -> str:
    norm = normalize_expr(expr)
    try:
        tree = ast.parse(norm, mode="eval")
    except SyntaxError as exc:
        raise ValueError("无法解析的算式") from exc
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED:
            raise ValueError("算式包含不允许的操作")
    try:
        value = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {})
    except ZeroDivisionError as exc:
        raise ValueError("除数不能为零") from exc
    except OverflowError as exc:
        raise ValueError("计算结果溢出") from exc
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def detect_arithmetic(query: str) -> Optional[str]:
    """若查询包含算术表达式则返回归一化算式，否则返回 None。

    从问句中抽取「由数字、运算符、括号、小数点、空格组成的连续片段」，
    校验为合法且仅含四则运算的求值表达式后返回。
    """
    norm = normalize_expr(query)
    for m in re.finditer(r"[0-9+\-*/().\s]+", norm):
        cand = m.group(0).strip()
        if not cand or not re.search(r"\d", cand) or not re.search(r"[+\-*/]", cand):
            continue
        try:
            tree = ast.parse(cand, mode="eval")
        except SyntaxError:
            continue
        if any(type(node) not in _ALLOWED for node in ast.walk(tree)):
            continue
        try:
            val = eval(compile(tree, "<c>", "eval"), {"__builtins__": {}}, {})
        except Exception:
            continue
        if isinstance(val, (int, float)):
            return cand
    return None


def now_chinese(offset_days: int = 0) -> str:
    d = datetime.date.today() + datetime.timedelta(days=offset_days)
    week = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][d.weekday()]
    return f"{d.year}年{d.month}月{d.day}日 {week}"


def detect_date(query: str) -> Optional[str]:
    q = query.lower()
    if any(k in query for k in ("今天", "明天", "昨天", "日期", "现在", "几号", "星期")) or "date" in q or "today" in q:
        if "明天" in query or "tomorrow" in q:
            return now_chinese(1)
        if "昨天" in query or "yesterday" in q:
            return now_chinese(-1)
        return now_chinese(0)
    return None


class SearchTool:
    """检索工具：封装 retriever，便于 Agent 调用与单测替换。"""

    def __init__(self, retriever: Callable[[str, int], List[RetrievedChunk]]) -> None:
        self._retriever = retriever

    def run(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        return self._retriever(query, top_k)
