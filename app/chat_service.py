"""对话服务（ch02）：DB 历史 → 模型定工具 → 执行 → 回灌收敛 → 流式回答。

约束驱动的两段式调用（ch01 文档核实：百炼兼容模式 tools 与 stream 不能同用）：
- 第一调：非流式 + bind_tools，让模型决定是否调工具；
- 有 tool_calls：推 tool 状态帧 → 逐个执行 → 申请单与结果落库 →
  第二调**换收敛专用 system**（不带工具使用压力，deepseek 等模型在强工具约束下
  会对"未找到"结果再次发起工具调用，见 dev-notes/ch02.md 阶段 8）→ 流式回灌；
- 收敛空流/异常 → 非流式重试一次 → 仍失败给兜底文案；最终 assistant 行**必定落库**，
  工具回合在历史里永远闭合；
- 无 tool_calls：第一调已有完整答案，分块伪流式吐出（省一次调用）。
全程 user/assistant(tool_calls)/tool/assistant(最终) 落 messages 表；
两次模型调用各有 asyncio.timeout 兜底，上游悬死不再拖死请求。
"""

import asyncio
import json
import logging

logger = logging.getLogger(__name__)

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.history import (
    drop_orphan_tool_prefix,
    repair_tool_rounds,
    rows_to_messages,
    trim_history,
)
from app.models import Conversation, Message
from app.prompts import CONVERGENCE_SYSTEM, REFUSAL_PREFIX, chat_prompt
from app.sse import citations_event, delta_event, done_event, error_event, meta_event, tool_event
from app.tools.base import ToolRegistry
from app.tokens import get_token_counter

FALLBACK_REPLY = "抱歉，刚才的查询没有取得有效结果，请您稍后重试，或让我帮您转接人工客服。"


class ChatService:
    def __init__(
        self,
        llm,
        session_factory: sessionmaker[Session],
        settings: Settings,
        registry_builder,
    ):
        self.llm = llm
        self.session_factory = session_factory
        self.settings = settings
        self.registry_builder = registry_builder
        self.count_tokens = get_token_counter()

    def _ensure_conversation(
        self, session: Session, user_id: str, conversation_id: int | None
    ) -> Conversation:
        if conversation_id is not None:
            conv = session.get(Conversation, conversation_id)
            if conv is not None:
                return conv
        conv = Conversation(user_id=user_id)
        session.add(conv)
        session.commit()
        return conv

    def _persist(self, rows: list[dict]) -> None:
        with self.session_factory() as session:
            for r in rows:
                session.add(Message(**r))
            session.commit()

    def _load_history(self, conversation_id: int) -> list[BaseMessage]:
        with self.session_factory() as session:
            rows = session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.id)
            ).all()
            return repair_tool_rounds(rows_to_messages(rows))

    def _persist_message(self, conversation_id: int, **fields) -> None:
        self._persist([{**fields, "conversation_id": conversation_id}])

    async def _pseudo_stream(self, text: str):
        """把一段完整文本分块吐成 delta（视觉与真流式一致）。"""
        for i in range(0, len(text), 3):
            yield delta_event(text[i : i + 3])
            await asyncio.sleep(0.02)

    async def stream_reply(
        self,
        user_id: str,
        user_message: str,
        conversation_id: int | None = None,
    ):
        try:
            with self.session_factory() as session:
                conv = self._ensure_conversation(session, user_id, conversation_id)
                conv_id = conv.id
                history = self._load_history(conv_id)
            self._persist_message(conv_id, role="user", content=user_message)
        except Exception as exc:
            yield error_event(f"数据库不可用: {type(exc).__name__}: {exc}")
            return

        yield meta_event(conv_id)

        window = drop_orphan_tool_prefix(
            trim_history(history, self.settings.history_token_budget, self.count_tokens)
        )
        messages = chat_prompt.invoke(
            {
                "platform_name": self.settings.platform_name,
                "history": window,
                "input": user_message,
            }
        ).to_messages()

        registry: ToolRegistry = self.registry_builder(self.session_factory, conv_id, self.llm)

        # ---- 第一调：非流式 + bind_tools（百炼 tools×stream 不同用，见 dev-notes） ----
        # 路由是决策型任务：temperature=0 保证选工具稳定
        try:
            async with asyncio.timeout(self.settings.llm_call_timeout_seconds):
                ai = await self.llm.bind_tools(
                    registry.all(), temperature=0
                ).ainvoke(messages)
        except Exception as exc:
            yield error_event(f"{type(exc).__name__}: {exc}")
            return

        if not ai.tool_calls:
            # 无工具轮：答案已完整，分块伪流式保持体验一致
            text = ai.text
            async for ev in self._pseudo_stream(text):
                yield ev
            if text:
                self._persist_message(conv_id, role="assistant", content=text)
            yield done_event()
            return

        # ---- 执行工具（状态帧先行；ch08 起统一走执行引擎——旧通道无确认 UI,写操作一律权限拒绝） ----
        from app.tools.engine import execute as engine_execute

        results = []
        for tc in ai.tool_calls:
            yield tool_event(tc["name"])
            out = await engine_execute(
                tc,
                registry,
                conversation_id=conv_id,
                session_factory=self.session_factory,
                default_timeout=self.settings.tool_timeout_seconds,
                max_retries=self.settings.tool_max_retries,
                allow_pending=False,
            )
            results.append(out.message)
        self._persist(
            [
                {
                    "conversation_id": conv_id,
                    "role": "assistant",
                    "content": ai.text or None,
                    "tool_calls": ai.tool_calls,
                }
            ]
            + [
                {
                    "conversation_id": conv_id,
                    "role": "tool",
                    "content": r.content,
                    "tool_call_id": r.tool_call_id,
                }
                for r in results
            ]
        )

        # ---- 第二调（收敛）：换专用 system，强制基于工具结果文本作答 ----
        conv_messages = (
            [SystemMessage(content=CONVERGENCE_SYSTEM.format(platform_name=self.settings.platform_name))]
            + messages[1:]
            + [ai]
            + results
        )

        # citations 帧：query_faq 的编号证据全集（答案角标 [n] → 来源 chunk）
        citations: list[dict] = []
        for r in results:
            try:
                data = json.loads(r.content)
                if isinstance(data, dict) and isinstance(data.get("matches"), list):
                    citations = data["matches"]
            except (json.JSONDecodeError, TypeError):
                continue
        if citations:
            yield citations_event(citations)

        full_reply = ""
        try:
            async with asyncio.timeout(self.settings.llm_call_timeout_seconds):
                async for chunk in self.llm.astream(conv_messages):
                    text = chunk.text
                    if text:
                        full_reply += text
                        yield delta_event(text)
        except Exception:
            full_reply = ""  # 超时/上游异常一律视为收敛失败，走重试兜底

        if not full_reply.strip():
            # 收敛失败（空流/异常）：非流式重试一次，仍失败给兜底文案
            retry_messages = conv_messages + [
                HumanMessage(content="（系统提示：不要再调用工具，请直接根据上方工具返回结果回答用户）")
            ]
            try:
                async with asyncio.timeout(self.settings.llm_call_timeout_seconds):
                    ai2 = await self.llm.ainvoke(retry_messages)
                full_reply = (ai2.text or "").strip()
            except Exception:
                full_reply = ""
            if not full_reply:
                full_reply = FALLBACK_REPLY
            async for ev in self._pseudo_stream(full_reply):
                yield ev

        # 拒答/低置信 → 低置信度问题池（best-effort，不影响主流）
        faq_no_hit = any("未检索到" in (r.content or "") for r in results)
        if full_reply.strip().startswith(REFUSAL_PREFIX):
            self._log_low_confidence(conv_id, user_message, "self_check", full_reply)
        elif faq_no_hit:
            self._log_low_confidence(
                conv_id, user_message, "retrieval_low_conf", "知识库检索未命中"
            )

        # 最终 assistant 行必定落库（工具回合在历史里永远闭合）
        self._persist_message(conv_id, role="assistant", content=full_reply)
        yield done_event()

    def _log_low_confidence(
        self, conversation_id: int, raw_question: str, source: str, reason: str
    ) -> None:
        try:
            from app.knowledge.models import LowConfidenceQuestion

            with self.session_factory() as session:
                session.add(
                    LowConfidenceQuestion(
                        conversation_id=conversation_id,
                        raw_question=raw_question,
                        source=source,
                        reason=reason[:500],
                    )
                )
                session.commit()
        except Exception:
            logger.exception("低置信度问题落池失败")  # 不拖垮对话主流程
