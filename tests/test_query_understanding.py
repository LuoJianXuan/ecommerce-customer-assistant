"""Query 理解接缝（TDD）：口语改写归一 + 同义词扩展（仅检索侧，入库不拆存）。"""

from app.knowledge.query_understanding import understand_query


class FakeLLM:
    def __init__(self, text):
        self._text = text
        self.seen = []

    def invoke(self, messages, config=None):
        self.seen.append([str(m.content) for m in messages])
        return self._text


def test_normalizes_colloquial_and_expands_synonyms():
    llm = FakeLLM('{"normalized": "退货寄回运费谁承担", "synonyms": ["退货运费", "寄回运费"]}')
    out = understand_query(llm, "我退回去的话那个邮费是你们出还是我出啊")
    assert out["normalized"] == "退货寄回运费谁承担"
    assert out["synonyms"] == ["退货运费", "寄回运费"]
    assert out["all_texts"][0] == "退货寄回运费谁承担"  # dense 用主查询


def test_invalid_json_falls_back_to_raw():
    llm = FakeLLM("抱歉我不太明白 JSON 要求")
    out = understand_query(llm, "邮费多少")
    assert out["normalized"] == "邮费多少"
    assert out["synonyms"] == []
    assert out["all_texts"] == ["邮费多少"]


def test_markdown_fenced_json_is_cleaned():
    llm = FakeLLM('```json\n{"normalized": "开发票流程", "synonyms": []}\n```')
    out = understand_query(llm, "发票咋搞")
    assert out["normalized"] == "开发票流程"


def test_synonyms_dedupe_and_capped():
    llm = FakeLLM(
        '{"normalized": "q", "synonyms": ["a", "a", "b", "c", "d", "e", "f"]}'
    )
    out = understand_query(llm, "x")
    assert len(out["synonyms"]) <= 5 and len(set(out["synonyms"])) == len(out["synonyms"])
