"""token 计数：优先 tiktoken（近似），不可用时回退字符启发式。

计数器以 Callable[[str], int] 注入使用方（如 trim_history），保证裁剪逻辑可独立单测。
"""

import math
from collections.abc import Callable

TokenCounter = Callable[[str], int]


def heuristic_token_count(text: str) -> int:
    """ASCII 约 4 字符折 1 token，非 ASCII（中日韩等）约 1 字 1 token。"""
    ascii_count = sum(1 for ch in text if ch.isascii())
    non_ascii_count = len(text) - ascii_count
    return math.ceil(ascii_count / 4) + non_ascii_count


def get_token_counter() -> TokenCounter:
    """tiktoken 可用则用 cl100k_base 近似（对 qwen 系模型是近似值），失败回退启发式。"""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")

        def tiktoken_count(text: str) -> int:
            return len(enc.encode(text))

        return tiktoken_count
    except Exception:
        return heuristic_token_count
