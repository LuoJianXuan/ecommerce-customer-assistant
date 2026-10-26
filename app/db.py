"""数据库接入：SQLAlchemy 2.0 + pymysql，连接参数来自 .env。

表结构以 sql/schema.sql 为准（应用不建表），模型仅做映射。
"""

from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, get_settings


def build_database_url(settings: Settings) -> str:
    return (
        f"mysql+pymysql://{settings.db_user}:{quote_plus(settings.db_password)}"
        f"@{settings.db_host}:{settings.db_port}/{settings.db_name}?charset=utf8mb4"
    )


# 进程级单例（Settings 不可哈希，不能用 lru_cache 按参数缓存）
_engine = None
_session_factory: sessionmaker[Session] | None = None


def get_engine(settings: Settings | None = None):
    global _engine
    if _engine is None:
        settings = settings or get_settings()
        _engine = create_engine(
            build_database_url(settings),
            pool_pre_ping=True,
            pool_recycle=3600,
        )
    return _engine


def get_session_factory(settings: Settings | None = None) -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(settings), expire_on_commit=False)
    return _session_factory


def get_db() -> Session:
    """FastAPI 依赖：请求级 session。"""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
