"""HTTP 接缝测试：/health 与聊天页首页。"""


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_reports_database_outage(client, monkeypatch):
    class BrokenEngine:
        def connect(self):
            raise OSError("database unavailable")

    monkeypatch.setattr("app.main.get_engine", lambda: BrokenEngine())

    resp = client.get("/health")

    assert resp.status_code == 503
    assert resp.json() == {"status": "degraded", "database": "unavailable"}


def test_root_serves_chat_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    body = resp.text
    # 页面骨架关键元素：标题、对话容器、输入框、发送按钮
    assert "小云" in body
    assert 'id="messages"' in body
    assert 'id="input"' in body
    assert 'id="send"' in body
