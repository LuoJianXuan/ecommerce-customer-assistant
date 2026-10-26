"""离线建库 CLI：Markdown 文档 → 切分 → MySQL(pending) → Milvus → 回填 done。

用法：
    python -m app.knowledge.build                      # 全量：knowledge_docs/ 全部文档
    python -m app.knowledge.build --resume             # 只补漏：扫 pending 向量化（验收2）
    python -m app.knowledge.build --docs 路径/*.md     # 指定文档
"""

import argparse
import sys
from pathlib import Path

from app.knowledge.embeddings import BgeM3Embedder
from app.knowledge.repository import replace_source, vectorize_pending
from app.knowledge.splitter import MarkdownKnowledgeSplitter
from app.knowledge.vector_store import MilvusKnowledgeStore

DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge_docs"


def build_documents(doc_paths: list[Path], content_type: str = "policy") -> int:
    """全量建库：逐文档 replace_source（同文档重导=整体替换）+ 双写向量化。"""
    from app.db import get_session_factory

    factory = get_session_factory()
    store = MilvusKnowledgeStore()
    store.ensure_collection()
    embedder = BgeM3Embedder.get()
    splitter = MarkdownKnowledgeSplitter(content_type=content_type)

    total = 0
    for path in doc_paths:
        source = path.stem
        chunks = splitter.split(path.read_text(encoding="utf-8"))
        with factory() as session:
            replace_source(session, source, chunks, store=store)
            session.commit()
        print(f"[build] {source}: {len(chunks)} 块落库(pending)")
        total += len(chunks)

    done = vectorize_pending(factory, embedder, store)
    print(f"[build] 向量化完成 {done} 块")
    return total


def resume_vectorization() -> int:
    """只补漏：把 pending 的块补向量化和回填（中断重跑，验收2）。"""
    from app.db import get_session_factory

    factory = get_session_factory()
    store = MilvusKnowledgeStore()
    store.ensure_collection()
    done = vectorize_pending(factory, BgeM3Embedder.get(), store)
    print(f"[resume] 补齐 {done} 块")
    return done


def rebuild_collection() -> int:
    """schema 升级用：drop 重建集合（chinese analyzer/BM25 函数建表定死）+ 全量重灌。

    MySQL 是权威源，全部块打回 pending 后走 --resume 同一链路；确定性 vector_id
    （kb-{chunk_id}）保证重建后主键不变。
    """
    from app.db import get_session_factory
    from app.knowledge.models import KnowledgeChunk

    factory = get_session_factory()
    store = MilvusKnowledgeStore()
    store.recreate()
    with factory() as session:
        n = (
            session.query(KnowledgeChunk)
            .where(KnowledgeChunk.governance_status == "active")
            .update(
                {KnowledgeChunk.vectorize_status: "pending", KnowledgeChunk.vector_id: None},
                synchronize_session=False,
            )
        )
        session.commit()
    print(f"[rebuild] 集合已重建，{n} 块打回 pending")
    return resume_vectorization()


def main() -> int:
    parser = argparse.ArgumentParser(description="知识库离线建库")
    parser.add_argument("--docs", nargs="*", default=None, help="Markdown 文档路径，默认 knowledge_docs/ 全部")
    parser.add_argument("--resume", action="store_true", help="只补漏向量化的 pending 块")
    parser.add_argument("--rebuild", action="store_true", help="drop 重建 Milvus 集合并全量重灌（schema 升级用）")
    args = parser.parse_args()

    if args.rebuild:
        return rebuild_collection()
    if args.resume:
        return resume_vectorization()
    paths = [Path(p) for p in args.docs] if args.docs else sorted(DOCS_DIR.glob("*.md"))
    if not paths:
        print(f"no docs found under {DOCS_DIR}")
        return 1
    return build_documents(paths)


if __name__ == "__main__":
    sys.exit(main())
