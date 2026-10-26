import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.config import Settings, get_settings
from app.db import get_engine
from app.llm import get_chat_model
from app.main import create_app, get_registry_builder
from tests.fakes import FakeChatLLM


@pytest.fixture(scope="session")
def db_ready():
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@pytest.fixture
def require_db(db_ready):
    if not db_ready:
        pytest.skip("MySQL 容器不可用")


@pytest.fixture
def fake_llm() -> FakeChatLLM:
    return FakeChatLLM()


def make_client(
    fake_llm,
    settings: Settings | None = None,
    registry=None,
) -> TestClient:
    """带假模型依赖注入的测试客户端（可选覆写 settings / 工具注册表）。"""
    app = create_app()
    app.dependency_overrides[get_chat_model] = lambda: fake_llm
    if settings is not None:
        app.dependency_overrides[get_settings] = lambda: settings
    if registry is not None:
        app.dependency_overrides[get_registry_builder] = (
            lambda: (lambda session_factory, conversation_id, llm=None: registry)
        )
    return TestClient(app)


@pytest.fixture
def client(fake_llm) -> TestClient:
    return make_client(fake_llm)
