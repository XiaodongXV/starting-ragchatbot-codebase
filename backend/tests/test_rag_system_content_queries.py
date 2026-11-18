"""Task 3 — how RAGSystem handles content-related questions end to end.

Real RAGSystem, real tools, real Chroma + embeddings (seeded tmp store); only the LLM is scripted.
So a failure here is a bug in OUR orchestration, not in the provider.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

import ai_generator
import rag_system as rag_module
from conftest import BACKEND_DIR, COMPUTER_USE, MCP
from document_processor import DocumentProcessor
from fakes import ScriptedLLM, make_response, make_tool_call
from rag_system import RAGSystem

SEARCH_CALL = make_tool_call(
    "search_course_content",
    {"query": "mouse clicks and screenshots", "course_name": "Computer Use"}, "call_1")


@pytest.fixture
def make_rag(monkeypatch, tmp_store):
    """Factory: RAGSystem wired to the seeded tmp store, LLM replaced by a ScriptedLLM."""
    def _make(*responses, max_history=2):
        monkeypatch.setattr(rag_module, "VectorStore", lambda *a, **k: tmp_store)
        llm = ScriptedLLM(*responses)
        monkeypatch.setattr(ai_generator.litellm, "completion", llm)
        cfg = SimpleNamespace(
            CHUNK_SIZE=800, CHUNK_OVERLAP=100, CHROMA_PATH="unused", EMBEDDING_MODEL="unused",
            MAX_RESULTS=5, MAX_RESULT_DISTANCE=1.2, LLM_API_KEY="", LLM_MODEL="gemini/test-model",
            LLM_FALLBACK_MODELS=["gemini/fallback"], MAX_HISTORY=max_history)
        return RAGSystem(cfg), llm
    return _make


# --------------------------------------------------------------------------------------
# Content question: tool call -> Chroma -> answer + sources
# --------------------------------------------------------------------------------------
class TestContentQuestion:
    def test_returns_answer_and_sources(self, make_rag):
        rag, _ = make_rag(make_response(tool_calls=[SEARCH_CALL]),
                          make_response(content="It lets the model click and type."))
        answer, sources = rag.query("How does computer use work?")
        assert answer == "It lets the model click and type."
        # course filter only (no lesson filter): the best-matching lesson comes first
        assert sources[0] == {"label": f"{COMPUTER_USE} - Lesson 0", "url": "https://example.com/cu/0"}
        assert all(s["label"].startswith(COMPUTER_USE) for s in sources)

    def test_prompt_wrapper_reaches_llm(self, make_rag):
        rag, llm = make_rag(make_response(content="x"))
        rag.query("How does computer use work?")
        user_msg = llm.calls[0]["messages"][1]["content"]
        assert user_msg == "Answer this question about course materials: How does computer use work?"

    def test_both_tools_offered_to_llm(self, make_rag):
        rag, llm = make_rag(make_response(content="x"))
        rag.query("q")
        names = [t["function"]["name"] for t in llm.calls[0]["tools"]]
        assert names == ["search_course_content", "get_course_outline"]

    def test_retrieved_chunk_is_what_the_llm_sees_in_second_call(self, make_rag):
        rag, llm = make_rag(make_response(tool_calls=[SEARCH_CALL]), make_response(content="ok"))
        rag.query("How does computer use work?")
        tool_msg = llm.calls[1]["messages"][-1]
        assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "call_1"
        assert "mouse clicks and keystrokes" in tool_msg["content"]
        assert f"[{COMPUTER_USE} - Lesson 0]" in tool_msg["content"]

    def test_sources_are_reset_after_each_query(self, make_rag):
        rag, _ = make_rag(make_response(tool_calls=[SEARCH_CALL]), make_response(content="ok"),
                          make_response(content="general answer"))
        _, first = rag.query("How does computer use work?")
        assert first
        _, second = rag.query("What is 2+2?")
        assert second == []
        assert rag.tool_manager.get_last_sources() == []

    def test_search_without_hits_gives_answer_and_no_sources(self, make_rag):
        call = make_tool_call("search_course_content", {"query": "x", "course_name": "MCP", "lesson_number": 9})
        rag, llm = make_rag(make_response(tool_calls=[call]), make_response(content="Nothing found."))
        answer, sources = rag.query("What is in MCP lesson 9?")
        assert answer == "Nothing found." and sources == []
        assert llm.calls[1]["messages"][-1]["content"].startswith("No relevant content found")

    def test_answer_is_never_none(self, make_rag):
        """H2 at orchestration level: provider returns empty content -> must still be a str."""
        rag, _ = make_rag(make_response(tool_calls=[SEARCH_CALL]),
                          make_response(content=None, finish_reason="length"))
        answer, _ = rag.query("How does computer use work?")
        assert isinstance(answer, str) and answer.strip()

    def test_bad_tool_arguments_from_model_do_not_crash_query(self, make_rag):
        bad = make_tool_call("search_course_content", {"query": "x", "course": "oops"})
        rag, _ = make_rag(make_response(tool_calls=[bad]), make_response(content="recovered"))
        answer, _ = rag.query("q")
        assert answer == "recovered"


class TestOutlineAndGeneralQuestions:
    def test_outline_question_uses_outline_tool_and_returns_lesson_sources(self, make_rag):
        call = make_tool_call("get_course_outline", {"course_name": "Computer Use"})
        rag, llm = make_rag(make_response(tool_calls=[call]), make_response(content="outline"))
        answer, sources = rag.query("What lessons are in the Computer Use course?")
        assert answer == "outline"
        assert "Lesson 0: Introduction" in llm.calls[1]["messages"][-1]["content"]
        assert [s["label"] for s in sources] == [
            COMPUTER_USE, "Lesson 0: Introduction", "Lesson 1: Overview of model capabilities"]

    def test_general_question_uses_no_tool_and_has_no_sources(self, make_rag):
        rag, llm = make_rag(make_response(content="4"))
        answer, sources = rag.query("What is 2+2?")
        assert (answer, sources) == ("4", [])
        assert len(llm.calls) == 1


# --------------------------------------------------------------------------------------
# Session handling
# --------------------------------------------------------------------------------------
class TestSessions:
    def test_exchange_stored_with_raw_user_query(self, make_rag):
        rag, _ = make_rag(make_response(content="a1"))
        sid = rag.session_manager.create_session()
        rag.query("question one", sid)
        assert rag.session_manager.get_conversation_history(sid) == "User: question one\nAssistant: a1"

    def test_history_is_passed_into_next_system_prompt(self, make_rag):
        rag, llm = make_rag(make_response(content="a1"), make_response(content="a2"))
        sid = rag.session_manager.create_session()
        rag.query("question one", sid)
        rag.query("and a follow-up?", sid)
        system = llm.calls[1]["messages"][0]["content"]
        assert "Previous conversation:" in system and "question one" in system and "a1" in system

    def test_history_truncated_to_max_history(self, make_rag):
        rag, llm = make_rag(make_response(content="a1"), make_response(content="a2"),
                            make_response(content="a3"), max_history=1)
        sid = rag.session_manager.create_session()
        for q in ("q1", "q2", "q3"):
            rag.query(q, sid)
        history = rag.session_manager.get_conversation_history(sid)
        assert "q3" in history and "q2" not in history and "q1" not in history

    def test_no_session_means_nothing_stored(self, make_rag):
        rag, llm = make_rag(make_response(content="a"))
        rag.query("q")
        assert rag.session_manager.sessions == {}
        assert "Previous conversation" not in llm.calls[0]["messages"][0]["content"]


# --------------------------------------------------------------------------------------
# Failure behaviour (characterisation of what the API layer sees)
# --------------------------------------------------------------------------------------
class TestFailurePropagation:
    def test_llm_error_propagates_to_caller(self, make_rag):
        """Characterisation: provider errors are NOT caught in RAGSystem; app.py turns them into
        HTTP 500 -> the frontend's 'Query failed'."""
        rag, _ = make_rag(RuntimeError("provider 400"))
        with pytest.raises(RuntimeError, match="provider 400"):
            rag.query("q")

    def test_error_in_second_call_propagates_and_sources_are_not_leaked(self, make_rag):
        rag, _ = make_rag(make_response(tool_calls=[SEARCH_CALL]), RuntimeError("provider 400 on follow-up"))
        with pytest.raises(RuntimeError):
            rag.query("How does computer use work?")
        # the failed query's sources must not bleed into the next query
        assert rag.tool_manager.get_last_sources() == []


class TestAnalytics:
    def test_course_analytics(self, make_rag):
        rag, _ = make_rag()
        a = rag.get_course_analytics()
        assert a["total_courses"] == 2
        assert set(a["course_titles"]) == {COMPUTER_USE, MCP}


# --------------------------------------------------------------------------------------
# Ingestion of the real docs/ folder (parsing only — no embeddings, no store writes)
# --------------------------------------------------------------------------------------
DOCS = sorted((BACKEND_DIR.parent / "docs").glob("*.txt"))


@pytest.mark.skipif(not DOCS, reason="no docs/*.txt")
@pytest.mark.parametrize("doc", DOCS, ids=[d.name for d in DOCS])
def test_real_course_documents_parse_into_searchable_chunks(doc: Path):
    course, chunks = DocumentProcessor(800, 100).process_course_document(str(doc))
    assert course.title and not course.title.lower().startswith("course title:")
    assert course.lessons, f"{doc.name}: no lessons parsed"
    assert chunks, f"{doc.name}: no chunks produced"
    assert all(c.course_title == course.title for c in chunks)
    assert all(c.lesson_number is not None for c in chunks)
    ids = [f"{c.course_title.replace(' ', '_')}_{c.chunk_index}" for c in chunks]
    assert len(ids) == len(set(ids)), "duplicate chunk ids would make Chroma raise"
