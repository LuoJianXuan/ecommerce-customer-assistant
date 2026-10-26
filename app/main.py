"""FastAPI 应用工厂与路由。"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Response
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from app.chat_service import ChatService
from app.config import Settings, get_settings
from app.conversations_api import router as conversations_router
from app.db import get_db, get_engine, get_session_factory
from app.agent.api import router as agent_router
from app.eval_api import router as eval_router
from app.extract_service import ExtractService
from app.kb_api import kb_page, router as kb_router
from app.review_api import router as review_router
from app.topics.api import router as topics_router
from app.llm import get_chat_model
from app.schemas import AfterSaleInfo, ChatRequest, ExtractRequest
from app.tools import build_default_registry

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

INDEX_HTML = Path(__file__).resolve().parent.parent / "static" / "index.html"
TOPIC_HTML = Path(__file__).resolve().parent.parent / "static" / "topic.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    import logging

    from app.knowledge.embeddings import BgeM3Embedder
    from app.knowledge.scheduler import start_mining_scheduler, stop_mining_scheduler
    from app.eval_scheduler import start_eval_scheduler, stop_eval_scheduler
    from app.topics.scheduler import start_topic_scheduler, stop_topic_scheduler
    from app.logging_setup import setup_file_logging

    setup_file_logging(get_settings().log_dir)  # ch07：model_ctx/history_ctx/summary 落 log/app.log

    async def _warmup_models() -> None:
        """顺序预热两个大模型，避免并发反序列化造成瞬时内存峰值。"""
        try:
            await asyncio.to_thread(BgeM3Embedder.get)  # 冷启动载模型 30-60s，别占事件循环
            logging.getLogger(__name__).info("BGE-M3 预热完成")
        except Exception:
            logging.getLogger(__name__).exception("BGE-M3 预热失败")
        try:
            from app.knowledge.reranker import BgeReranker

            await asyncio.to_thread(BgeReranker.get)  # ch04：重排模型 2.27GB 同样预热
            logging.getLogger(__name__).info("BGE-Reranker 预热完成")
        except Exception:
            logging.getLogger(__name__).exception("BGE-Reranker 预热失败")

    warmup_task = asyncio.create_task(_warmup_models())
    # ch07：上下文预算启动自检（连一轮都装不下就报警）
    from app.config import get_settings as _gs
    from app.context_budget import budget_from_settings, check_budget

    _budget = budget_from_settings(_gs())
    _log = logging.getLogger("context")
    for _alarm in check_budget(_budget, _gs().ctx_steady_per_round):
        _log.error(_alarm)
    _log.info(
        "[context-budget] 滑窗=%s 层1=%s 层2=%s（窗口侧=%s 固定开销=%s 峰值=%s）",
        _budget.history, _budget.layer1, _budget.layer2,
        _budget.window_side, _budget.fixed_overhead, _budget.peak,
    )
    start_mining_scheduler()
    start_eval_scheduler()
    start_topic_scheduler()
    # ch09：Langfuse 环境变量在首个 CallbackHandler 创建前注入（时序教训同 HF——
    # SDK 客户端按 env 初始化，晚了就静默 disabled）；停机时 flush 缓冲区防丢 trace
    _lf = _gs()
    if _lf.langfuse_enabled and _lf.langfuse_public_key:
        import os
        from urllib.parse import urlparse

        os.environ["LANGFUSE_PUBLIC_KEY"] = _lf.langfuse_public_key
        os.environ["LANGFUSE_SECRET_KEY"] = _lf.langfuse_secret_key
        os.environ["LANGFUSE_HOST"] = _lf.langfuse_host
        # ch08 老坑复用：系统代理劫持 127.0.0.1，OTLP 上传器吃代理 env——把 Langfuse
        # 主机名并进 NO_PROXY（大小写两种都设，SDK/导出器读法不一）
        _lf_host = urlparse(_lf.langfuse_host).hostname or "127.0.0.1"
        for _k in ("NO_PROXY", "no_proxy"):
            _cur = os.environ.get(_k, "")
            _parts = [p.strip() for p in _cur.split(",") if p.strip()]
            if _lf_host not in _parts:
                _parts.append(_lf_host)
            os.environ[_k] = ",".join(_parts)
        logging.getLogger(__name__).info("Langfuse 观测已启用 host=%s", _lf.langfuse_host)
    yield
    warmup_task.cancel()
    stop_mining_scheduler()
    stop_eval_scheduler()
    stop_topic_scheduler()
    if _lf.langfuse_enabled and _lf.langfuse_public_key:
        try:
            from langfuse import get_client

            get_client().shutdown()  # flush 在途 trace（默认批量上传）
        except Exception:
            logging.getLogger(__name__).exception("Langfuse 停机 flush 失败（不阻塞关服）")


def get_registry_builder():
    """返回 (session_factory, conversation_id) -> ToolRegistry 的构建器；测试可覆写。"""
    return build_default_registry


def create_app() -> FastAPI:
    app = FastAPI(title="电商智能客服", version="0.3.0", lifespan=lifespan)

    @app.get("/")
    def index() -> FileResponse:
        """聊天页面（单页，官方推荐 FileResponse 而非 mount StaticFiles）。"""
        return FileResponse(INDEX_HTML)

    app.include_router(kb_router)
    app.include_router(eval_router)
    app.include_router(agent_router)
    app.include_router(conversations_router)
    app.include_router(review_router)
    app.include_router(topics_router)

    @app.get("/kb")
    def knowledge_base_page():
        """知识库管理页面。"""
        return kb_page()

    @app.get("/eval")
    def eval_page():
        """评估与编造台账页面。"""
        from fastapi.responses import FileResponse
        from pathlib import Path as _P
        return FileResponse(_P(__file__).resolve().parent.parent / "static" / "eval.html")

    @app.get("/review")
    def review_page():
        """知识缺口审核与评估趋势页面。"""
        return FileResponse(
            Path(__file__).resolve().parent.parent / "static" / "review.html"
        )

    @app.get("/topic")
    def topic_page():
        """低置信问题主题分布与批量归类页面。"""
        return FileResponse(TOPIC_HTML)

    @app.get("/health")
    def health(response: Response) -> dict:
        try:
            with get_engine().connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception:
            response.status_code = 503
            return {"status": "degraded", "database": "unavailable"}
        return {"status": "ok"}

    @app.post("/api/chat")
    async def chat(
        req: ChatRequest,
        llm=Depends(get_chat_model),
        settings: Settings = Depends(get_settings),
        registry_builder=Depends(get_registry_builder),
        _db: Session = Depends(get_db),
    ) -> StreamingResponse:
        svc = ChatService(
            llm,
            get_session_factory(settings),
            settings,
            registry_builder,
        )
        return StreamingResponse(
            svc.stream_reply(req.user_id, req.message, req.conversation_id),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    @app.post("/api/after-sale/extract", response_model=AfterSaleInfo)
    async def extract(
        req: ExtractRequest, llm=Depends(get_chat_model)
    ) -> AfterSaleInfo:
        return await ExtractService(llm).extract(req.text)

    return app


app = create_app()
