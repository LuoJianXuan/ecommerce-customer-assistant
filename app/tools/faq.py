"""FAQ 工具（ch04：混合检索+重排+Query 理解；契约仍为 keyword 入/JSON 出）。

入参出参契约：keyword: str → {"keyword":..., "matches":[...]}。
ch04 起 matches 条目带编号 n（供模型角标引用与前端 citations 映射），
并保留 section_path/chunk_id（溯源）。score 只记日志。
执行超时按工具覆盖（改写+检索+重排总预算 FAQ_TOOL_TIMEOUT_SECONDS）。
"""

import json
import logging

from sqlalchemy.orm import Session, sessionmaker
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def make_query_faq(
    session_factory: sessionmaker[Session], llm=None, conversation_id: int | None = None
):
    @tool
    def query_faq(keyword: str) -> str:
        """按语义检索平台知识库（FAQ/退货政策/售后手册/会员说明等），返回带编号的最相关问答内容。

        用户问政策、规则、费用、流程类问题（如退货政策、邮费运费、发票、保修、会员权益、付款方式）时调用；
        请把用户的完整问题原句传进来（语义检索，不要求与库内字面一致）。

        Args:
            keyword: 用户的完整问题或查询语句，如 "邮费是多少"
        """
        from app.config import get_settings
        from app.knowledge.query_understanding import understand_query
        from app.knowledge.retrieval import reorder_for_prompt, retrieve

        settings = get_settings()
        query_texts = None
        if settings.query_rewrite_enabled and llm is not None:
            # Query 理解（检索侧）：口语改写归一 + 同义词扩展；失败降级原查询
            query_texts = understand_query(llm, keyword)["all_texts"]

        matches = retrieve(
            session_factory,
            keyword,
            top_k=settings.rerank_top_k,
            strategy=settings.retrieval_strategy,
            query_texts=query_texts,
        )
        from app.knowledge.flywheel import save_retrieval_snapshot

        save_retrieval_snapshot(session_factory, conversation_id, keyword, matches)
        if not matches:
            return f"知识库中未检索到与「{keyword}」相关的内容"
        # 首尾组装（最相关放首尾，缓解 lost-in-the-middle），再编号
        ordered = reorder_for_prompt(matches)
        logger.info(
            "query_faq hit: %s",
            [(m["question"][:16], round(m["score"] or 0, 3)) for m in ordered[:5]],
        )
        return json.dumps(
            {
                "keyword": keyword,
                "matches": [
                    {
                        "n": i + 1,
                        "question": m["question"],
                        "answer": m["answer"],
                        "category": m["category"],
                        "section_path": m.get("section_path"),
                        "chunk_id": m.get("chunk_id"),
                    }
                    for i, m in enumerate(ordered)
                ],
            },
            ensure_ascii=False,
        )

    return query_faq
