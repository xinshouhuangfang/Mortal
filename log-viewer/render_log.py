#!/usr/bin/env python3
"""Render a mahjong log into a self-contained viewer page.

Usage:
    python render_log.py <log.json> [--output index.html]

Reads the template `index.example.html` (next to this script), replaces the
embedded `allActions` block with the given log file (one JSON event per line),
and writes the result so it can be opened directly in a browser via file://.
"""

import argparse
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_FILE = SCRIPT_DIR / "index.example.html"
DEFAULT_OUTPUT = SCRIPT_DIR / "index.html"

ALL_ACTIONS_RE = re.compile(r"(allActions = `)[^`]*(`)", re.DOTALL)


def read_log_lines(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        sys.exit(f"错误: {path} 是 gzip 压缩文件(.json.gz)。请先用未压缩的 .json,例如:\n"
                 f"  gzip -dk {path}")
    text = raw.decode("utf-8")
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        sys.exit(f"错误: {path} 为空文件")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="把牌谱渲染进 log-viewer 页面")
    parser.add_argument("log", type=Path, help="未压缩的牌谱文件路径(.json,每行一条 JSON 事件)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"输出文件(默认 {DEFAULT_OUTPUT.name})")
    args = parser.parse_args()

    if not TEMPLATE_FILE.exists():
        sys.exit(f"错误: 找不到模板 {TEMPLATE_FILE}")
    if not args.log.exists():
        sys.exit(f"错误: 文件不存在 {args.log}")

    template = TEMPLATE_FILE.read_text(encoding="utf-8")
    content = read_log_lines(args.log)

    new_html, n = ALL_ACTIONS_RE.subn(rf"\g<1>{content}\g<2>", template, count=1)
    if n != 1:
        sys.exit("错误: 模板里找不到 allActions 块,模板可能已被改动")

    args.output.write_text(new_html, encoding="utf-8")
    print(f"已生成: {args.output}")
    print(f"用浏览器打开 {args.output.name} 即可回放 {args.log}")


if __name__ == "__main__":
    main()
