"""P0 门禁：全仓字符扫描，检测是否将 emoji 用作功能图标。

作者：晨星
仅用 ord() 码点范围判定，源文件内绝不出现 emoji 字面量，避免被 GBK 代码页破坏。
返回 0 表示无 emoji；返回 1 表示发现 emoji（阻断交付）。
"""

from __future__ import annotations

import os
import sys

# emoji / 符号码点区间（不含 CJK 0x4E00-0x9FFF）
_RANGES = [
    (0x1F300, 0x1F9FF), (0x2600, 0x26FF), (0x2700, 0x27BF),
    (0xFE00, 0xFE0F), (0x1F000, 0x1F02F), (0x1F0A0, 0x1F0FF),
    (0x1F100, 0x1F64F), (0x1F680, 0x1F6FF), (0x1F900, 0x1F9FF),
    (0x1FA00, 0x1FA6F), (0x1FA70, 0x1FAFF), (0x200D, 0x200D),
    (0x20E3, 0x20E3), (0xE0020, 0xE007F),
]


def _is_emoji(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _RANGES)


def scan(root: str, extensions=(".py",)) -> list:
    findings = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if not name.endswith(extensions):
                continue
            path = os.path.join(dirpath, name)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    for ln, line in enumerate(fh, 1):
                        for ch in line:
                            if _is_emoji(ch):
                                findings.append((path, ln, ch))
            except Exception:
                continue
    return findings


def main() -> int:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    findings = scan(root)
    if findings:
        for path, ln, ch in findings:
            rel = os.path.relpath(path, root)
            print(f"[P0-EMOJI] {rel}:{ln} 发现符号 U+{ord(ch):04X}")
        print(f"发现 {len(findings)} 处 emoji / 符号，违反 P0 绝对规则")
        return 1
    print("P0 字符门禁通过：未发现 emoji / 符号用作功能图标")
    return 0


if __name__ == "__main__":
    sys.exit(main())
