"""/api/kb/* 知识库管理接口 + /kb 页面路由。

复用 ch03 核心能力（splitter/retrieval/build/faq_import/mine），
只包 API 壳；长任务（建库/导入/挖矿）走线程后台执行 + 内存任务表轮询。
不做登录鉴权/审计/多人协同（本地单人后台，计划已声明）。
"""

import threading
import uuid
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app.db import get_session_factory
from app.knowledge.models import KnowledgeChunk, QaExtractionStaging
from app.knowledge.splitter import MarkdownKnowledgeSplitter

KB_HTML = Path(__file__).resolve().parent.parent / "static" / "kb.html"

router = APIRouter(prefix="/api/kb", tags=["knowledge-base"])

# ---------- 长任务：线程执行 + 内存状态表 ----------

_jobs: dict[str, dict] = {}
_job_lock = threading.Lock()


def _run_job(kind: str, runner: Callable[[], object]) -> None:
    try:
        result = runner()
        with _job_lock:
            _jobs[_current_id[kind]]["status"] = "done"
            _jobs[_current_id[kind]]["result"] = result
    except Exception as exc:
        with _job_lock:
            _jobs[_current_id[kind]]["status"] = "failed"
            _jobs[_current_id[kind]]["result"] = f"{type(exc).__name__}: {exc}"
    finally:
        _current_id.pop(kind, None)


_current_id: dict[str, str] = {}


def _launch_job(kind: str, label: str, runner: Callable[[], object]) -> dict:
    with _job_lock:
        if kind in _current_id:
            raise HTTPException(409, f"任务 {label} 已在运行")
        job_id = uuid.uuid4().hex[:8]
        _jobs[job_id] = {"id": job_id, "kind": kind, "label": label, "status": "running"}
        _current_id[kind] = job_id
    threading.Thread(target=_run_job, args=(kind, runner), daemon=True).start()
    return _jobs[job_id]


class PreviewRequest(BaseModel):
    markdown: str = Field(min_length=1, max_length=200_000)


class IngestRequest(PreviewRequest):
    title: str = Field(min_length=1, max_length=64)


@router.get("/stats")
def stats() -> dict:
    factory = get_session_factory()
    milvus_ok = True
    try:
        from app.knowledge.vector_store import MilvusKnowledgeStore

        MilvusKnowledgeStore().client.list_collections()
    except Exception:
        milvus_ok = False
    with factory() as s:
        rows = s.execute(
            select(KnowledgeChunk.vectorize_status, func.count()).group_by(
                KnowledgeChunk.vectorize_status
            )
        ).all()
        staging = s.execute(
            select(QaExtractionStaging.status, func.count()).group_by(
                QaExtractionStaging.status
            )
        ).all()
    by_status = {k: v for k, v in rows}
    by_staging = {k: v for k, v in staging}
    return {
        "total": sum(by_status.values()),
        "done": by_status.get("done", 0),
        "pending": by_status.get("pending", 0),
        "staging_kept": by_staging.get("kept", 0),
        "staging_discarded": by_staging.get("discarded", 0),
        "staging_extracted": by_staging.get("extracted", 0),
        "milvus_ok": milvus_ok,
    }


@router.get("/chunks")
def list_chunks(
    q: str = "",
    category: str = "",
    status: str = "",
    source: str = "",
    limit: int = Query(default=50, le=200),
) -> dict:
    factory = get_session_factory()
    with factory() as s:
        stmt = select(KnowledgeChunk).order_by(KnowledgeChunk.id.desc()).limit(limit)
        if q:
            like = f"%{q}%"
            stmt = stmt.where(
                KnowledgeChunk.questions.like(like) | KnowledgeChunk.answer.like(like)
            )
        if category:
            stmt = stmt.where(KnowledgeChunk.category == category)
        if status:
            stmt = stmt.where(KnowledgeChunk.vectorize_status == status)
        if source:  # 材料维度：md 文档名前缀或 FAQ知识库
            stmt = stmt.where(KnowledgeChunk.section_path.like(f"{source}%"))
        rows = s.scalars(stmt).all()
    return {
        "items": [
            {
                "id": r.id,
                "category": r.category,
                "questions": r.questions,
                "answer": r.answer,
                "section_path": r.section_path,
                "content_type": r.content_type,
                "is_key_clause": bool(r.is_key_clause),
                "vectorize_status": r.vectorize_status,
            }
            for r in rows
        ]
    }


@router.get("/categories")
def categories() -> dict:
    factory = get_session_factory()
    with factory() as s:
        rows = s.execute(
            select(KnowledgeChunk.category, func.count())
            .group_by(KnowledgeChunk.category)
            .order_by(func.count().desc())
        ).all()
    return {"items": [{"category": c, "count": n} for c, n in rows]}


@router.post("/preview")
def preview(req: PreviewRequest) -> dict:
    chunks = MarkdownKnowledgeSplitter().split(req.markdown)
    # 查重提示：库内真实问法（归一化）被块内容覆盖（包含式）→ 疑似重复，不阻断
    from app.knowledge.mine import normalize_text

    factory = get_session_factory()
    with factory() as s:
        existing_qs = [normalize_text(q) for (q,) in s.execute(select(KnowledgeChunk.questions)).all()]
    existing_qs = [q for q in existing_qs if q]

    def _is_dup(chunk) -> bool:
        combined = normalize_text(chunk.questions + chunk.answer)
        return any(q in combined for q in existing_qs)

    dup_count = sum(1 for c in chunks if _is_dup(c))
    return {
        "count": len(chunks),
        "dup_count": dup_count,
        "chunks": [
            {
                "category": c.category,
                "questions": c.questions,
                "answer": c.answer,
                "section_path": c.section_path,
                "is_key_clause": bool(c.is_key_clause),
            }
            for c in chunks
        ],
    }


@router.get("/materials")
def materials() -> dict:
    """建库材料清单：knowledge_docs/*.md 就地切块干跑（不写库不碰 Milvus）+ faq 表聚合。"""
    from app.knowledge.build import DOCS_DIR
    from app.models import Faq

    splitter = MarkdownKnowledgeSplitter()
    items, total = [], 0
    for p in sorted(DOCS_DIR.glob("*.md")):
        text = p.read_text(encoding="utf-8")
        chunks = splitter.split(text)
        has_table = "|---" in text
        sections = len({c.section_path for c in chunks})
        stem = p.stem
        mtype = "manual" if "手册" in stem else ("faq" if "faq" in stem.lower() else "policy")
        feats = ("有表 · " if has_table else "") + f"{sections} 节"
        items.append(
            {
                "file": p.name,
                "type": mtype,
                "chars": len(text),
                "lines": text.count("\n") + 1,
                "chunks": len(chunks),
                "key_clauses": sum(1 for c in chunks if c.is_key_clause),
                "features": feats,
            }
        )
        total += len(chunks)

    factory = get_session_factory()
    with factory() as s:
        faq_rows = s.scalars(select(Faq)).all()
    if faq_rows:
        chars = sum(len(r.question) + len(r.answer) for r in faq_rows)
        items.append(
            {
                "file": "faq 表（MySQL）",
                "type": "faq",
                "chars": chars,
                "lines": len(faq_rows),
                "chunks": len(faq_rows),
                "key_clauses": 0,
                "features": f"{len(faq_rows)} 条问答",
            }
        )
        total += len(faq_rows)
    return {"items": items, "count": len(items), "total_chunks": total}


@router.get("/mining")
def mining() -> dict:
    """对话挖知识：暂存三态 + 挖矿次数（批次时间戳去重）+ 最近暂存条目。"""
    factory = get_session_factory()
    with factory() as s:
        rows = s.execute(
            select(QaExtractionStaging.status, func.count()).group_by(QaExtractionStaging.status)
        ).all()
        runs = s.scalar(
            select(func.count(func.distinct(func.substring(QaExtractionStaging.batch_no, 6, 14))))
        ) or 0
        recent = s.scalars(
            select(QaExtractionStaging).order_by(QaExtractionStaging.id.desc()).limit(10)
        ).all()
    by = {k: v for k, v in rows}
    return {
        "extracted": by.get("extracted", 0),
        "kept": by.get("kept", 0),
        "discarded": by.get("discarded", 0),
        "runs": runs,
        "recent": [
            {
                "id": r.id,
                "batch_no": r.batch_no,
                "question": r.question,
                "answer": r.answer[:60],
                "status": r.status,
            }
            for r in recent
        ],
    }


@router.get("/dualwrite")
def dualwrite() -> dict:
    """向量化与双写：pending/done + Milvus 实数 + 最近账本。"""
    from app.config import get_settings
    from app.knowledge.vector_store import MilvusKnowledgeStore

    factory = get_session_factory()
    with factory() as s:
        rows = s.execute(
            select(KnowledgeChunk.vectorize_status, func.count()).group_by(
                KnowledgeChunk.vectorize_status
            )
        ).all()
        ledger_rows = s.scalars(
            select(KnowledgeChunk).order_by(KnowledgeChunk.updated_at.desc(), KnowledgeChunk.id.desc()).limit(15)
        ).all()
    by = {k: v for k, v in rows}
    milvus_count = 0
    try:
        st = MilvusKnowledgeStore().client.get_collection_stats(get_settings().milvus_collection)
        milvus_count = int(st.get("row_count", 0))
    except Exception:
        milvus_count = -1  # 不可达标记，前端显示 ✗
    return {
        "pending": by.get("pending", 0),
        "done": by.get("done", 0),
        "milvus_count": milvus_count,
        "collection": get_settings().milvus_collection,
        "ledger": [
            {
                "id": r.id,
                "questions": r.questions[:24],
                "vectorize_status": r.vectorize_status,
                "vector_id": r.vector_id,
                "updated_at": r.updated_at.strftime("%m-%d %H:%M") if r.updated_at else "",
            }
            for r in ledger_rows
        ],
    }


@router.post("/ingest")
def ingest(req: IngestRequest, background: BackgroundTasks) -> dict:
    chunks = MarkdownKnowledgeSplitter().split(req.markdown)
    if not chunks:
        raise HTTPException(422, "文档切不出任何块（缺标题或正文为空）")

    def runner() -> dict:
        from app.knowledge.embeddings import BgeM3Embedder
        from app.knowledge.repository import replace_source, vectorize_pending
        from app.knowledge.vector_store import MilvusKnowledgeStore

        factory = get_session_factory()
        store = MilvusKnowledgeStore()
        store.ensure_collection()
        with factory() as s:
            replace_source(s, req.title, chunks, store=store)
            s.commit()
        done = vectorize_pending(factory, BgeM3Embedder.get(), store)
        return {"chunks": len(chunks), "vectorized": done}

    job = _launch_job("ingest", f"录入「{req.title}」", runner)
    return {"job": job, "chunks": len(chunks)}


@router.post("/jobs/{kind}")
def start_job(kind: str) -> dict:
    if kind == "build-docs":
        def runner() -> dict:
            from app.knowledge.build import DOCS_DIR, build_documents

            paths = sorted(DOCS_DIR.glob("*.md"))
            total = build_documents(paths)
            return {"docs": len(paths), "chunks": total}

        return _launch_job(kind, "重建文档库", runner)
    if kind == "import-faq":
        def runner() -> dict:
            from app.knowledge.faq_import import import_faq

            return {"vectorized": import_faq()}

        return _launch_job(kind, "导入 FAQ", runner)
    if kind == "vectorize-pending":
        def runner() -> dict:
            from app.knowledge.build import resume_vectorization

            return {"resumed": resume_vectorization()}

        return _launch_job(kind, "补齐向量化", runner)
    if kind == "mine":
        def runner() -> dict:
            import asyncio

            from app.knowledge.mine import run_mining

            return asyncio.run(run_mining())

        return _launch_job(kind, "挖一轮知识", runner)
    raise HTTPException(422, f"未知任务类型: {kind}")


@router.get("/jobs")
def job_status() -> dict:
    with _job_lock:
        return {"items": list(_jobs.values())[-20:]}


@router.get("/search")
def search(q: str, top_k: int = Query(default=3, le=10), strategy: str = "") -> dict:
    from app.config import get_settings
    from app.knowledge.retrieval import retrieve

    if strategy and strategy not in {"dense", "bm25", "hybrid", "hybrid_rerank"}:
        raise HTTPException(422, f"未知检索策略: {strategy}")
    hits = retrieve(
        get_session_factory(),
        q,
        top_k=top_k,
        strategy=strategy or get_settings().retrieval_strategy,
    )
    return {
        "hits": [
            {
                "question": h["question"],
                "answer": h["answer"],
                "category": h["category"],
                "section_path": h.get("section_path"),
                "chunk_id": h.get("chunk_id"),
                "score": round(h["score"], 4) if h.get("score") is not None else None,
            }
            for h in hits
        ]
    }


def kb_page() -> FileResponse:
    return FileResponse(KB_HTML)
