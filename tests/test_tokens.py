"""token 计数接缝：启发式计数（手算样例为独立真值）。

BPE 的典型行为：ASCII 文本约 4 字符折 1 token，中日韩字符约 1 字 1 token。
"""

from app.tokens import heuristic_token_count


def test_empty_text_is_zero_tokens():
    assert heuristic_token_count("") == 0


def test_ascii_about_four_chars_per_token():
    # "hello" 5 个 ASCII 字符 -> ceil(5/4) = 2
    assert heuristic_token_count("hello") == 2


def test_cjk_one_token_per_char():
    # "你好世界" 4 个汉字 -> 4
    assert heuristic_token_count("你好世界") == 4


def test_mixed_text_sums_both_parts():
    # "hi你好"：ASCII "hi" -> ceil(2/4)=1，汉字 2 个 -> 2，合计 3
    assert heuristic_token_count("hi你好") == 3
