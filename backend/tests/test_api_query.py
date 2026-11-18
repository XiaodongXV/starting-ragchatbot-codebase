"""POST /api/query — the exact place where 'Query failed' is born.

frontend/script.js: `if (!response.ok) throw new Error('Query failed')`
backend/app.py:     any exception in rag_system.query() -> HTTPException(500)
"""
import importlib
import sys
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import rag_system as rag_module
from conftest import BACKEND_DIR


@pytest.fixture
def api(monkeypatch):
    """app.py with RAGSystem replaced by a MagicMock (app builds it at import time)."""
    fake_rag = MagicMock()
    fake_rag.session_manager.create_session.return_value = "session_1"
    fake_rag.query.return_value = ("the answer", [{"label": "C - Lesson 1", "url": None}])
    monkeypatch.setattr(rag_module, "RAGSystem", lambda cfg: fake_rag)
    monkeypatch.chdir(BACKEND_DIR)               # app mounts StaticFiles("../frontend") relative to cwd
    sys.modules.pop("app", None)
    app_module = importlib.import_module("app")
    yield TestClient(app_module.app), fake_rag
    sys.modules.pop("app", None)


def test_success_shape_and_session_creation(api):
    client, rag = api
    r = client.post("/api/query", json={"query": "what is MCP?"})
    assert r.status_code == 200
    assert r.json() == {"answer": "the answer",
                        "sources": [{"label": "C - Lesson 1", "url": None}],
                        "session_id": "session_1"}
    rag.query.assert_called_once_with("what is MCP?", "session_1")


def test_existing_session_is_reused(api):
    client, rag = api
    client.post("/api/query", json={"query": "q", "session_id": "session_9"})
    rag.session_manager.create_session.assert_not_called()
    rag.query.assert_called_once_with("q", "session_9")


def test_exception_in_rag_becomes_500_with_detail(api):
    """Characterisation: this is what the user sees as 'Query failed'; the real cause is only in `detail`."""
    client, rag = api
    rag.query.side_effect = RuntimeError("provider 400: function call is missing thought_signature")
    r = client.post("/api/query", json={"query": "q"})
    assert r.status_code == 500
    assert "thought_signature" in r.json()["detail"]


def test_exception_in_rag_is_logged_with_traceback(api, caplog):
    """The frontend only shows 'Query failed', so the real cause must reach the server log."""
    client, rag = api
    rag.query.side_effect = RuntimeError("provider 503 overloaded")
    with caplog.at_level("ERROR", logger="uvicorn.error"):
        client.post("/api/query", json={"query": "what is MCP?"})
    record = next(r for r in caplog.records if "/api/query failed" in r.getMessage())
    assert "what is MCP?" in record.getMessage()
    assert record.exc_info and "provider 503 overloaded" in str(record.exc_info[1])


def test_none_answer_becomes_500(api):
    """Characterisation of H2: `QueryResponse.answer: str` rejects None -> 500."""
    client, rag = api
    rag.query.return_value = (None, [])
    assert client.post("/api/query", json={"query": "q"}).status_code == 500


def test_missing_query_is_422(api):
    client, _ = api
    assert client.post("/api/query", json={}).status_code == 422
