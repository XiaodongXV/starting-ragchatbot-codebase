"""Task 1 — behaviour of CourseSearchTool.execute (and the ToolManager that wraps it).

Three layers:
  A. mocked VectorStore       -> pure formatting / routing logic of the tool
  B. real seeded VectorStore  -> tool + real Chroma + real embeddings
  C. copy of the real index   -> the data the running app actually serves
"""
import pytest

from conftest import COMPUTER_USE, MCP, EMBEDDING_MODEL
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import SearchResults, VectorStore


# --------------------------------------------------------------------------------------
# A. Mocked store — logic inside the tool
# --------------------------------------------------------------------------------------
class TestToolDefinition:
    def test_definition_shape(self, mock_store):
        d = CourseSearchTool(mock_store).get_tool_definition()
        assert d["name"] == "search_course_content"
        assert d["description"]
        assert d["input_schema"]["type"] == "object"
        assert d["input_schema"]["required"] == ["query"]
        props = d["input_schema"]["properties"]
        assert set(props) == {"query", "course_name", "lesson_number"}
        assert props["lesson_number"]["type"] == "integer"


class TestExecuteRouting:
    def test_forwards_all_arguments_to_store(self, mock_store):
        CourseSearchTool(mock_store).execute("what is MCP", course_name="MCP", lesson_number=2)
        mock_store.search.assert_called_once_with(query="what is MCP", course_name="MCP", lesson_number=2)

    def test_optional_arguments_default_to_none(self, mock_store):
        CourseSearchTool(mock_store).execute("what is MCP")
        mock_store.search.assert_called_once_with(query="what is MCP", course_name=None, lesson_number=None)

    def test_store_error_is_returned_verbatim(self, mock_store):
        mock_store.search.return_value = SearchResults.empty("Search error: boom")
        assert CourseSearchTool(mock_store).execute("x") == "Search error: boom"

    def test_no_course_found_error_is_returned(self, mock_store):
        mock_store.search.return_value = SearchResults.empty("No course found matching 'zzz'")
        assert "No course found matching 'zzz'" in CourseSearchTool(mock_store).execute("x", course_name="zzz")


class TestEmptyResults:
    @pytest.fixture(autouse=True)
    def _empty(self, mock_store):
        mock_store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])

    def test_plain_empty_message(self, mock_store):
        assert CourseSearchTool(mock_store).execute("x") == "No relevant content found."

    def test_message_mentions_course_and_lesson(self, mock_store):
        out = CourseSearchTool(mock_store).execute("x", course_name="MCP", lesson_number=3)
        assert out == "No relevant content found in course 'MCP' in lesson 3."

    def test_message_mentions_lesson_zero(self, mock_store):
        """Lesson 0 is a real lesson in the docs ('Lesson 0: Introduction'); `if lesson_number:` drops it."""
        out = CourseSearchTool(mock_store).execute("x", lesson_number=0)
        assert "lesson 0" in out


class TestFormatting:
    def test_each_chunk_has_course_and_lesson_header(self, mock_store):
        out = CourseSearchTool(mock_store).execute("computer use")
        assert f"[{COMPUTER_USE} - Lesson 1]\nchunk A about computer use" in out
        assert f"[{MCP} - Lesson 2]\nchunk C about MCP" in out

    def test_chunk_without_lesson_has_header_without_lesson(self, mock_store):
        mock_store.search.return_value = SearchResults(
            documents=["orphan"], metadata=[{"course_title": "C"}], distances=[0.1])
        assert CourseSearchTool(mock_store).execute("x").startswith("[C]\norphan")

    def test_missing_course_title_is_labelled_unknown(self, mock_store):
        mock_store.search.return_value = SearchResults(documents=["d"], metadata=[{}], distances=[0.1])
        assert "[unknown]" in CourseSearchTool(mock_store).execute("x")


class TestSources:
    def test_sources_deduplicated_per_course_and_lesson(self, mock_store):
        tool = CourseSearchTool(mock_store)
        tool.execute("computer use")
        # 3 chunks, but two belong to (COMPUTER_USE, 1)
        assert [s["label"] for s in tool.last_sources] == [
            f"{COMPUTER_USE} - Lesson 1", f"{MCP} - Lesson 2"]

    def test_sources_carry_lesson_url(self, mock_store):
        tool = CourseSearchTool(mock_store)
        tool.execute("computer use")
        assert tool.last_sources[0]["url"] == "https://example.com/1"
        mock_store.get_lesson_link.assert_any_call(COMPUTER_USE, 1)

    def test_source_without_lesson_has_no_url(self, mock_store):
        mock_store.search.return_value = SearchResults(
            documents=["d"], metadata=[{"course_title": "C"}], distances=[0.1])
        tool = CourseSearchTool(mock_store)
        tool.execute("x")
        assert tool.last_sources == [{"label": "C", "url": None}]

    def test_sources_do_not_go_stale_after_empty_search(self, mock_store):
        """A later search that finds nothing must not leave the previous search's sources behind."""
        tool = CourseSearchTool(mock_store)
        tool.execute("computer use")
        assert tool.last_sources
        mock_store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])
        tool.execute("nothing here")
        assert tool.last_sources == []


# --------------------------------------------------------------------------------------
# B. Real VectorStore on seeded data
# --------------------------------------------------------------------------------------
class TestExecuteAgainstRealStore:
    @pytest.fixture
    def tool(self, tmp_store):
        return CourseSearchTool(tmp_store)

    def test_unfiltered_query_returns_relevant_chunk(self, tool):
        out = tool.execute("how does the model operate a computer with mouse clicks")
        assert f"[{COMPUTER_USE} - Lesson 0]" in out
        assert "mouse clicks" in out

    def test_partial_course_name_is_resolved(self, tool):
        out = tool.execute("protocol for tools", course_name="MCP")
        assert f"[{MCP} - Lesson 1]" in out
        assert COMPUTER_USE not in out  # filter really applied

    def test_lesson_filter_restricts_results(self, tool):
        out = tool.execute("model", lesson_number=1)
        assert "Lesson 0" not in out
        assert "Lesson 1" in out

    def test_course_and_lesson_filter_combined(self, tool):
        out = tool.execute("multimodal images", course_name="Computer Use", lesson_number=1)
        assert f"[{COMPUTER_USE} - Lesson 1]" in out
        assert "multimodal" in out

    def test_lesson_zero_is_searchable(self, tool):
        out = tool.execute("screenshot", course_name="Computer Use", lesson_number=0)
        assert f"[{COMPUTER_USE} - Lesson 0]" in out

    def test_lesson_with_no_content_reports_empty(self, tool):
        out = tool.execute("anything", course_name="MCP", lesson_number=9)
        assert out.startswith("No relevant content found")

    def test_sources_populated_with_real_links(self, tool):
        tool.execute("multimodal images", course_name="Computer Use", lesson_number=1)
        assert tool.last_sources == [
            {"label": f"{COMPUTER_USE} - Lesson 1", "url": "https://example.com/cu/1"}]

    def test_unknown_course_is_reported_as_not_found(self, tool):
        """`_resolve_course_name` always returns the nearest course (no distance threshold),
        so a nonsense course name silently searches the wrong course instead of saying 'not found'."""
        out = tool.execute("recipes", course_name="Italian Pasta Cooking Masterclass")
        assert "No course found" in out


class TestOutlineTool:
    def test_outline_lists_all_lessons(self, tmp_store):
        out = CourseOutlineTool(tmp_store).execute("Computer Use")
        assert f"Course: {COMPUTER_USE}" in out
        assert "Lessons (2):" in out
        assert "Lesson 0: Introduction" in out
        assert "Lesson 1: Overview of model capabilities" in out

    def test_outline_sources(self, tmp_store):
        tool = CourseOutlineTool(tmp_store)
        tool.execute("Computer Use")
        assert tool.last_sources[0] == {"label": COMPUTER_USE, "url": "https://example.com/computer-use"}
        assert len(tool.last_sources) == 3


# --------------------------------------------------------------------------------------
# ToolManager — what the AI generator actually calls
# --------------------------------------------------------------------------------------
class TestToolManager:
    @pytest.fixture
    def manager(self, mock_store):
        m = ToolManager()
        m.register_tool(CourseSearchTool(mock_store))
        m.register_tool(CourseOutlineTool(mock_store))
        return m

    def test_both_tools_registered(self, manager):
        assert {d["name"] for d in manager.get_tool_definitions()} == {
            "search_course_content", "get_course_outline"}

    def test_dispatches_by_name(self, manager, mock_store):
        out = manager.execute_tool("search_course_content", query="q")
        assert "chunk A" in out
        mock_store.search.assert_called_once()

    def test_unknown_tool_returns_message(self, manager):
        assert "not found" in manager.execute_tool("nope", query="q")

    def test_unexpected_argument_from_model_does_not_raise(self, manager):
        """LLMs sometimes add keys that are not in the schema. That must degrade to an error
        string the model can read, not a TypeError that becomes HTTP 500."""
        out = manager.execute_tool("search_course_content", query="q", unexpected_key="x")
        assert isinstance(out, str)

    def test_missing_required_argument_does_not_raise(self, manager):
        out = manager.execute_tool("search_course_content")
        assert isinstance(out, str)

    def test_store_exception_does_not_raise(self, manager, mock_store):
        mock_store.search.side_effect = RuntimeError("chroma exploded")
        out = manager.execute_tool("search_course_content", query="q")
        assert isinstance(out, str) and "chroma exploded" in out

    def test_get_and_reset_sources(self, manager):
        manager.execute_tool("search_course_content", query="q")
        assert manager.get_last_sources()
        manager.reset_sources()
        assert manager.get_last_sources() == []


# --------------------------------------------------------------------------------------
# C. The real index that the running app serves (copied, never mutated)
# --------------------------------------------------------------------------------------
class TestRealIndex:
    @pytest.fixture
    def real_store(self, real_db_copy):
        return VectorStore(real_db_copy, EMBEDDING_MODEL, max_results=5)

    def test_index_is_populated(self, real_store):
        assert real_store.get_course_count() > 0, "course_catalog is empty — nothing was ever indexed"
        assert real_store.course_content.count() > 0, "course_content is empty"

    def test_every_course_has_metadata_and_chunks(self, real_store):
        for title in real_store.get_existing_course_titles():
            meta = real_store.course_catalog.get(ids=[title])["metadatas"][0]
            assert meta["title"] == title
            assert meta["lessons_json"] not in ("", "[]"), f"{title}: no lessons stored"
            chunks = real_store.course_content.get(where={"course_title": title}, limit=1)
            assert chunks["ids"], f"{title}: catalog entry exists but no content chunks"

    def test_content_search_returns_results(self, real_store):
        res = real_store.search("computer use mouse clicks and keystrokes")
        assert res.error is None
        assert not res.is_empty()
        assert {"course_title", "lesson_number", "chunk_index"} <= set(res.metadata[0])

    @pytest.mark.parametrize("name, expected_start", [
        ("MCP", "MCP:"),                       # short acronym: semantic distance alone (1.55) would reject it
        ("Computer Use", "Building Towards Computer Use"),
        ("computer use", "Building Towards Computer Use"),
        ("Chroma", "Advanced Retrieval"),
    ])
    def test_real_partial_course_names_resolve(self, real_store, name, expected_start):
        assert real_store._resolve_course_name(name).startswith(expected_start)

    @pytest.mark.parametrize("name", ["Italian Pasta Cooking Masterclass", "zzz", "Quantum Physics"])
    def test_nonsense_course_names_do_not_resolve(self, real_store, name):
        assert real_store._resolve_course_name(name) is None

    def test_tool_end_to_end_on_real_data(self, real_store):
        tool = CourseSearchTool(real_store)
        out = tool.execute("take a screenshot and click", course_name="Computer Use")
        assert not out.startswith(("Search error", "No course found", "No relevant content"))
        assert out.startswith("[")
        assert tool.last_sources
