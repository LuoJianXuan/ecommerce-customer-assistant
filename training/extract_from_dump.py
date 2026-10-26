"""从备份 mysqldump 提取 low_confidence_questions 真实问题 → training/data/real_questions.jsonl。

真实问题在本章只作"真实回归评估集",不进训练集(计划第 0 节已确认决策)。
Docker 环境恢复后,新数据改由 prepare_corpus.py 直连 DB 捞取,本脚本作为留痕的可复现入口。

用法:
    python training/extract_from_dump.py [dump路径]
        必须显式传入私有 dump 文件路径；导出内容不进入公开仓库。
"""

import json
import re
import sys
from pathlib import Path

OUT = Path(__file__).parent / "data" / "real_questions.jsonl"

# low_confidence_questions 列序(以 dump 内 CREATE TABLE 为准)
COLUMNS = [
    "id",
    "conversation_id",
    "raw_question",
    "source",
    "reason",
    "created_at",
    "retrieval_snapshot",
    "matched_review_id",
]

STRING_ESCAPES = {
    "\\": "\\",
    "'": "'",
    '"': '"',
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "0": "\0",
}


def _parse_values(values: str) -> list[list]:
    """逐字符解析 mysqldump extended insert 的 VALUES 列表。

    处理:单引号字符串 + 反斜杠转义 + NULL + 数字,忽略注释性内容。
    """
    rows: list[list] = []
    row: list = []
    i, n = 0, len(values)
    while i < n:
        ch = values[i]
        if ch in ",)" and not row and ch == ")":
            i += 1
            continue
        if ch == "(":
            row = []
            i += 1
            continue
        if ch == ")":
            rows.append(row)
            row = []
            i += 1
            continue
        if ch == ",":
            i += 1
            continue
        if values.startswith("NULL", i):
            row.append(None)
            i += 4
            continue
        if ch == "'":
            i += 1
            buf: list[str] = []
            while i < n:
                c = values[i]
                if c == "\\" and i + 1 < n:
                    buf.append(STRING_ESCAPES.get(values[i + 1], values[i + 1]))
                    i += 2
                elif c == "'":
                    if i + 1 < n and values[i + 1] == "'":  # '' 转义兜底
                        buf.append("'")
                        i += 2
                    else:
                        i += 1
                        break
                else:
                    buf.append(c)
                    i += 1
            row.append("".join(buf))
            continue
        # 数字(含负号/小数)
        m = re.match(r"-?\d+(?:\.\d+)?", values[i:])
        if m:
            tok = m.group(0)
            row.append(float(tok) if "." in tok else int(tok))
            i += len(tok)
            continue
        i += 1  # 跳过空白等
    return rows


def extract(dump_path: Path) -> list[dict]:
    text = dump_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(
        r"INSERT INTO `low_confidence_questions` VALUES (.+?);\n",
        text,
        re.S,
    )
    if not m:
        raise SystemExit(f"dump 中未找到 low_confidence_questions 的 INSERT: {dump_path}")
    rows = _parse_values(m.group(1))
    out = []
    for r in rows:
        rec = dict(zip(COLUMNS, r))
        out.append(
            {
                "id": rec["id"],
                "conversation_id": rec["conversation_id"],
                "raw_question": rec["raw_question"],
                "source": rec["source"],
                "reason": (rec["reason"] or "")[:200],
                "created_at": str(rec["created_at"]),
            }
        )
    return out


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("用法: python training/extract_from_dump.py <private-dump.sql>")
    dump = Path(sys.argv[1])
    rows = extract(dump)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"提取 {len(rows)} 条 → {OUT}")


if __name__ == "__main__":
    main()
