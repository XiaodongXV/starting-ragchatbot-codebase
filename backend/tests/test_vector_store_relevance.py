"""Relevance cut-off and exclude_course in VectorStore.search.

Real Chroma + real embeddings on the seeded tmp store (2 tiny courses). The store is session-wide
and shared, so each test sets `max_distance` through monkeypatch and leaves it as it found it.
"""
import pytest

from conftest import COMPUTER_USE, MCP
from search_tools import CourseSearchTool
from vector_store import SearchResults

ON_TOPIC = "mouse clicks and keystrokes on a computer"
OFF_TOPIC = "Italian pasta recipes"


class TestSearchResultsWithin:
    def test_keeps_only_chunks_within_distance_and_stays_aligned(self):
        res = SearchResults(documents=["a", "b", "c"],
                            metadata=[{"i": 1}, {"i": 2}, {"i": 3}],
                            distances=[0.4, 1.3, 1.0])
        kept = res.within(1.0)
        assert kept.documents == ["a", "c"]
        assert kept.metadata == [{"i": 1}, {"i": 3}]
        assert kept.distances == [0.4, 1.0]

    def test_everything_too_far_gives_empty_results(self):
        assert SearchResults(documents=["a"], metadata=[{}], distances=[2.0]).within(1.0).is_empty()


class TestRelevanceCutoff:
    def test_off_topic_open_search_returns_nothing(self, tmp_store, monkeypatch):
        monkeypatch.setattr(tmp_store, "max_distance", 1.2)
        assert tmp_store.search(OFF_TOPIC).is_empty()

    def test_on_topic_open_search_still_returns_results(self, tmp_store, monkeypatch):
        monkeypatch.setattr(tmp_store, "max_distance", 1.2)
        res = tmp_store.search(ON_TOPIC)
        assert not res.is_empty()
        assert res.metadata[0]["course_title"] == COMPUTER_USE

    def test_no_cutoff_configured_keeps_old_behaviour(self, tmp_store, monkeypatch):
        monkeypatch.setattr(tmp_store, "max_distance", None)
        assert not tmp_store.search(OFF_TOPIC).is_empty()

    @pytest.mark.parametrize("kwargs", [{"course_name": "Computer Use"}, {"lesson_number": 0}])
    def test_cutoff_does_not_apply_when_user_named_a_course_or_lesson(self, tmp_store, monkeypatch, kwargs):
        """Naming a course/lesson means 'whatever is there'; an empty answer would be wrong."""
        monkeypatch.setattr(tmp_store, "max_distance", 0.1)
        assert not tmp_store.search(OFF_TOPIC, **kwargs).is_empty()

    def test_tool_reports_nothing_relevant_for_off_topic_question(self, tmp_store, monkeypatch):
        monkeypatch.setattr(tmp_store, "max_distance", 1.2)
        assert CourseSearchTool(tmp_store).execute(OFF_TOPIC) == "No relevant content found."


class TestExcludeCourse:
    def test_excluded_course_never_appears(self, tmp_store):
        res = tmp_store.search("tools and resources for language models", exclude_course="MCP")
        assert not res.is_empty()
        assert {m["course_title"] for m in res.metadata} == {COMPUTER_USE}

    def test_partial_name_is_resolved(self, tmp_store):
        res = tmp_store.search("anything about models", exclude_course="Computer Use")
        assert {m["course_title"] for m in res.metadata} == {MCP}

    def test_unknown_excluded_course_is_reported(self, tmp_store):
        res = tmp_store.search("anything", exclude_course="Italian Pasta Cooking Masterclass")
        assert res.error and "No course found matching" in res.error

    def test_exclude_works_together_with_lesson_filter(self, tmp_store):
        res = tmp_store.search("introduction", lesson_number=1, exclude_course="MCP")
        assert {m["course_title"] for m in res.metadata} == {COMPUTER_USE}
        assert {m["lesson_number"] for m in res.metadata} == {1}

    def test_tool_reports_nothing_outside_the_excluded_course(self, tmp_store, monkeypatch):
        monkeypatch.setattr(tmp_store, "max_distance", 0.5)
        out = CourseSearchTool(tmp_store).execute("Model Context Protocol tools", exclude_course="MCP")
        assert out == "No relevant content found outside course 'MCP'."


class TestBuildFilter:
    @pytest.fixture
    def build(self, tmp_store):
        return tmp_store._build_filter

    def test_no_arguments_means_no_filter(self, build):
        assert build(None, None) is None

    def test_single_clauses_are_not_wrapped_in_and(self, build):
        assert build("A", None) == {"course_title": "A"}
        assert build(None, 2) == {"lesson_number": 2}
        assert build(None, None, "B") == {"course_title": {"$ne": "B"}}

    def test_lesson_zero_is_a_real_filter(self, build):
        assert build(None, 0) == {"lesson_number": 0}

    def test_combined_clauses(self, build):
        assert build("A", 2, "B") == {"$and": [
            {"course_title": "A"}, {"lesson_number": 2}, {"course_title": {"$ne": "B"}}]}
