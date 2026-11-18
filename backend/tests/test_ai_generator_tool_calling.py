"""Task 2 — does AIGenerator correctly offer and invoke the search tool?

`litellm.completion` is replaced with a scripted fake, so these tests are offline and deterministic.
They check the wire contract (what we send) and the tool loop (what we do with the reply).
"""
import json
from unittest.mock import MagicMock

import pytest

import ai_generator
from ai_generator import AIGenerator
from fakes import ScriptedLLM, make_response, make_tool_call
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager


@pytest.fixture
def generator():
    # api_key="" -> constructor does not touch os.environ
    return AIGenerator("", "gemini/test-model", ["gemini/fallback-1", "gemini/fallback-2"])


@pytest.fixture
def real_tool_defs(mock_store):
    manager = ToolManager()
    manager.register_tool(CourseSearchTool(mock_store))
    manager.register_tool(CourseOutlineTool(mock_store))
    return manager.get_tool_definitions()


@pytest.fixture
def tool_manager():
    tm = MagicMock(spec=ToolManager)
    tm.execute_tool.return_value = "RESULT-FROM-TOOL"
    return tm


def install(monkeypatch, *responses) -> ScriptedLLM:
    llm = ScriptedLLM(*responses)
    monkeypatch.setattr(ai_generator.litellm, "completion", llm)
    return llm


SEARCH_ARGS = {"query": "what is computer use", "course_name": "Computer Use"}


# --------------------------------------------------------------------------------------
# First call: what we send to the model
# --------------------------------------------------------------------------------------
class TestFirstCall:
    def test_tools_are_sent_in_openai_format(self, generator, monkeypatch, real_tool_defs, tool_manager):
        llm = install(monkeypatch, make_response(content="hi"))
        generator.generate_response("q", tools=real_tool_defs, tool_manager=tool_manager)

        sent = llm.calls[0]["tools"]
        assert [t["function"]["name"] for t in sent] == ["search_course_content", "get_course_outline"]
        for t, original in zip(sent, real_tool_defs):
            assert t["type"] == "function"
            assert t["function"]["description"] == original["description"]
            assert t["function"]["parameters"] == original["input_schema"]
        assert llm.calls[0]["tool_choice"] == "auto"

    def test_model_params_and_fallbacks_forwarded(self, generator, monkeypatch, real_tool_defs, tool_manager):
        llm = install(monkeypatch, make_response(content="hi"))
        generator.generate_response("q", tools=real_tool_defs, tool_manager=tool_manager)
        call = llm.calls[0]
        assert call["model"] == "gemini/test-model"
        assert call["fallbacks"] == ["gemini/fallback-1", "gemini/fallback-2"]
        assert call["temperature"] == 0
        assert call["max_tokens"] > 0

    def test_messages_are_system_then_user(self, generator, monkeypatch):
        llm = install(monkeypatch, make_response(content="hi"))
        generator.generate_response("my question")
        msgs = llm.calls[0]["messages"]
        assert [m["role"] for m in msgs] == ["system", "user"]
        assert msgs[0]["content"] == AIGenerator.SYSTEM_PROMPT
        assert msgs[1]["content"] == "my question"

    def test_history_is_appended_to_system_prompt(self, generator, monkeypatch):
        llm = install(monkeypatch, make_response(content="hi"))
        generator.generate_response("q", conversation_history="User: earlier\nAssistant: answer")
        system = llm.calls[0]["messages"][0]["content"]
        assert system.startswith(AIGenerator.SYSTEM_PROMPT)
        assert "Previous conversation:\nUser: earlier\nAssistant: answer" in system

    def test_without_tools_no_tool_keys_are_sent(self, generator, monkeypatch):
        llm = install(monkeypatch, make_response(content="hi"))
        generator.generate_response("q")
        assert "tools" not in llm.calls[0]
        assert "tool_choice" not in llm.calls[0]

    def test_system_prompt_routes_content_and_outline_questions(self):
        p = AIGenerator.SYSTEM_PROMPT
        assert "search_course_content" in p and "get_course_outline" in p

    def test_system_prompt_guides_cross_course_search_and_forbids_outside_facts(self):
        """Text only: whether the model obeys is measured by backend/eval_answers.py."""
        p = AIGenerator.SYSTEM_PROMPT
        assert "exclude_course" in p and "WITHOUT `course_name`" in p
        assert "Only state what the tool results contain" in p

    def test_system_prompt_allows_two_sequential_tool_calls(self):
        p = AIGenerator.SYSTEM_PROMPT
        assert "One tool call per query" not in p
        assert "2 sequential tool calls" in p


# --------------------------------------------------------------------------------------
# Tool loop: does a tool_call from the model end up executing the right tool?
# --------------------------------------------------------------------------------------
class TestToolInvocation:
    def test_no_tool_call_returns_content_directly(self, generator, monkeypatch, real_tool_defs, tool_manager):
        llm = install(monkeypatch, make_response(content="Paris."))
        out = generator.generate_response("capital of France?", tools=real_tool_defs, tool_manager=tool_manager)
        assert out == "Paris."
        assert len(llm.calls) == 1
        tool_manager.execute_tool.assert_not_called()

    def test_search_tool_call_executes_with_parsed_arguments(self, generator, monkeypatch, real_tool_defs, tool_manager):
        install(monkeypatch,
                make_response(tool_calls=[make_tool_call("search_course_content", SEARCH_ARGS)]),
                make_response(content="Computer use is ..."))
        out = generator.generate_response("what is computer use?", tools=real_tool_defs, tool_manager=tool_manager)
        tool_manager.execute_tool.assert_called_once_with("search_course_content", **SEARCH_ARGS)
        assert out == "Computer use is ..."

    def test_outline_tool_call_is_routed_by_name(self, generator, monkeypatch, real_tool_defs, tool_manager):
        install(monkeypatch,
                make_response(tool_calls=[make_tool_call("get_course_outline", {"course_name": "MCP"})]),
                make_response(content="Outline ..."))
        generator.generate_response("outline of MCP?", tools=real_tool_defs, tool_manager=tool_manager)
        tool_manager.execute_tool.assert_called_once_with("get_course_outline", course_name="MCP")

    def test_integer_lesson_number_survives_json_roundtrip(self, generator, monkeypatch, real_tool_defs, tool_manager):
        install(monkeypatch,
                make_response(tool_calls=[make_tool_call(
                    "search_course_content", {"query": "x", "lesson_number": 2})]),
                make_response(content="ok"))
        generator.generate_response("q", tools=real_tool_defs, tool_manager=tool_manager)
        assert tool_manager.execute_tool.call_args.kwargs["lesson_number"] == 2

    def test_tool_calls_ignored_when_no_tool_manager(self, generator, monkeypatch, real_tool_defs):
        install(monkeypatch, make_response(content="text", tool_calls=[make_tool_call("search_course_content", SEARCH_ARGS)]))
        assert generator.generate_response("q", tools=real_tool_defs) == "text"

    def test_multiple_tool_calls_all_executed_and_answered(self, generator, monkeypatch, real_tool_defs, tool_manager):
        calls = [make_tool_call("search_course_content", {"query": "a"}, "id_a"),
                 make_tool_call("search_course_content", {"query": "b"}, "id_b")]
        llm = install(monkeypatch, make_response(tool_calls=calls), make_response(content="done"))
        generator.generate_response("q", tools=real_tool_defs, tool_manager=tool_manager)
        assert tool_manager.execute_tool.call_count == 2
        tool_msgs = [m for m in llm.calls[1]["messages"] if isinstance(m, dict) and m["role"] == "tool"]
        assert [m["tool_call_id"] for m in tool_msgs] == ["id_a", "id_b"]


# --------------------------------------------------------------------------------------
# Second call: what we send back after running the tool
# --------------------------------------------------------------------------------------
class TestFollowUpCall:
    @pytest.fixture
    def run(self, generator, monkeypatch, real_tool_defs, tool_manager):
        first = make_response(tool_calls=[make_tool_call("search_course_content", SEARCH_ARGS, "call_xyz")])
        llm = install(monkeypatch, first, make_response(content="final answer"))
        out = generator.generate_response("what is computer use?", tools=real_tool_defs, tool_manager=tool_manager)
        return llm, first, out

    def test_message_order_and_tool_call_id(self, run):
        llm, first, _ = run
        msgs = llm.calls[1]["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[1] == {"role": "user", "content": "what is computer use?"}
        assert msgs[2] is first.choices[0].message          # assistant turn that requested the tool
        assert msgs[3] == {"role": "tool", "tool_call_id": "call_xyz", "content": "RESULT-FROM-TOOL"}
        assert len(msgs) == 4

    def test_model_and_fallbacks_forwarded_on_second_call(self, run):
        llm, _, _ = run
        assert llm.calls[1]["model"] == "gemini/test-model"
        assert llm.calls[1]["fallbacks"] == ["gemini/fallback-1", "gemini/fallback-2"]

    def test_returns_text_of_second_call(self, run):
        assert run[2] == "final answer"

    def test_followup_call_still_allows_another_tool_call(self, run):
        """After the first tool result the model may still chain a second call (round 2 of 2)."""
        llm, _, _ = run
        assert llm.calls[1]["tool_choice"] == "auto"

    def test_followup_call_declares_tools_when_history_contains_tool_calls(self, run):
        """H1 contract check. The replayed history contains an assistant `tool_calls` turn plus a
        `tool` turn. Anthropic (and Gemini function-calling) reject such history when the request
        declares no tools. LiteLLM papers over this for Anthropic only; whether it is fatal for the
        active provider is decided by the live test."""
        llm, _, _ = run
        assert llm.calls[1].get("tools"), "follow-up request replays tool_calls but declares no tools"


# --------------------------------------------------------------------------------------
# Robustness — every exception here currently surfaces as HTTP 500 -> "Query failed"
# --------------------------------------------------------------------------------------
class TestRobustness:
    def test_empty_final_content_still_returns_a_string(self, generator, monkeypatch, real_tool_defs, tool_manager):
        """H2: if the model's thinking eats max_tokens, content is None; `QueryResponse.answer: str`
        then fails validation. generate_response must always hand back a str."""
        install(monkeypatch,
                make_response(tool_calls=[make_tool_call("search_course_content", SEARCH_ARGS)]),
                make_response(content=None, finish_reason="length"))
        out = generator.generate_response("q", tools=real_tool_defs, tool_manager=tool_manager)
        assert isinstance(out, str) and out.strip()

    def test_empty_direct_content_still_returns_a_string(self, generator, monkeypatch):
        install(monkeypatch, make_response(content=None, finish_reason="length"))
        out = generator.generate_response("q")
        assert isinstance(out, str) and out.strip()

    @pytest.mark.parametrize("bad_args", ["{not json", "", "[1, 2]", "null"])
    def test_malformed_tool_arguments_do_not_propagate(
            self, generator, monkeypatch, real_tool_defs, tool_manager, bad_args):
        install(monkeypatch,
                make_response(tool_calls=[make_tool_call("search_course_content", bad_args)]),
                make_response(content="recovered"))
        assert generator.generate_response("q", tools=real_tool_defs, tool_manager=tool_manager) == "recovered"

    def test_tool_exception_is_fed_back_to_model_not_raised(self, generator, monkeypatch, real_tool_defs, tool_manager):
        tool_manager.execute_tool.side_effect = TypeError("unexpected keyword argument 'foo'")
        llm = install(monkeypatch,
                      make_response(tool_calls=[make_tool_call("search_course_content", {"query": "q", "foo": 1}, "c1")]),
                      make_response(content="recovered"))
        out = generator.generate_response("q", tools=real_tool_defs, tool_manager=tool_manager)
        assert out == "recovered"
        tool_msg = llm.calls[1]["messages"][-1]
        assert tool_msg["tool_call_id"] == "c1" and "unexpected keyword" in tool_msg["content"]

    def test_tool_result_is_json_serialisable_message(self, generator, monkeypatch, real_tool_defs, tool_manager):
        """The follow-up payload (minus the provider's assistant object) must be plain JSON."""
        llm = install(monkeypatch,
                      make_response(tool_calls=[make_tool_call("search_course_content", SEARCH_ARGS)]),
                      make_response(content="ok"))
        generator.generate_response("q", tools=real_tool_defs, tool_manager=tool_manager)
        for m in llm.calls[1]["messages"]:
            if isinstance(m, dict):
                json.dumps(m)


# --------------------------------------------------------------------------------------
# Sequential tool calling: up to 2 rounds, each its own API request
# --------------------------------------------------------------------------------------
OUTLINE_CALL = make_tool_call("get_course_outline", {"course_name": "X"}, "call_outline")
SEARCH_CALL = make_tool_call("search_course_content", {"query": "Topic of lesson 4"}, "call_search")


def tool_messages(call):
    return [m for m in call["messages"] if isinstance(m, dict) and m["role"] == "tool"]


def empty_search_results():
    from vector_store import SearchResults
    return SearchResults(documents=[], metadata=[], distances=[])


class TestSequentialToolCalls:
    @pytest.fixture
    def chained(self, generator, monkeypatch, real_tool_defs, tool_manager):
        tool_manager.execute_tool.side_effect = ["Lesson 4: Topic of lesson 4", "Course Y covers it"]
        llm = install(monkeypatch,
                      make_response(tool_calls=[OUTLINE_CALL]),
                      make_response(tool_calls=[SEARCH_CALL]),
                      make_response(content="Course Y covers the same topic."))
        out = generator.generate_response("Which course matches lesson 4 of X?",
                                          conversation_history="User: hi\nAssistant: hello",
                                          tools=real_tool_defs, tool_manager=tool_manager)
        return llm, tool_manager, out

    def test_tools_run_in_order_and_final_text_is_returned(self, chained):
        llm, tm, out = chained
        assert [c.args[0] for c in tm.execute_tool.call_args_list] == ["get_course_outline", "search_course_content"]
        assert tm.execute_tool.call_args_list[1].kwargs == {"query": "Topic of lesson 4"}
        assert out == "Course Y covers the same topic."
        assert len(llm.calls) == 3

    def test_second_request_sees_first_tool_result(self, chained):
        llm, _, _ = chained
        assert [m["content"] for m in tool_messages(llm.calls[1])] == ["Lesson 4: Topic of lesson 4"]

    def test_third_request_sees_the_whole_exchange_in_order(self, chained):
        llm, _, _ = chained
        msgs = llm.calls[2]["messages"]
        assert [m["role"] if isinstance(m, dict) else m.role for m in msgs] == \
               ["system", "user", "assistant", "tool", "assistant", "tool"]
        assert [m["tool_call_id"] for m in tool_messages(llm.calls[2])] == ["call_outline", "call_search"]

    def test_history_is_kept_on_every_request(self, chained):
        llm, _, _ = chained
        for call in llm.calls:
            assert "User: hi\nAssistant: hello" in call["messages"][0]["content"]

    def test_tools_stay_available_for_both_rounds_then_are_forbidden(self, chained):
        llm, _, _ = chained
        assert [c["tool_choice"] for c in llm.calls] == ["auto", "auto", "none"]
        assert all(c.get("tools") for c in llm.calls)

    def test_fallbacks_forwarded_on_every_request(self, chained):
        llm, _, _ = chained
        assert all(c["fallbacks"] == ["gemini/fallback-1", "gemini/fallback-2"] for c in llm.calls)

    def test_answer_after_first_round_ends_without_final_none_request(
            self, generator, monkeypatch, real_tool_defs, tool_manager):
        llm = install(monkeypatch, make_response(tool_calls=[OUTLINE_CALL]), make_response(content="done"))
        assert generator.generate_response("q", tools=real_tool_defs, tool_manager=tool_manager) == "done"
        assert len(llm.calls) == 2
        assert tool_manager.execute_tool.call_count == 1

    def test_no_third_tool_round_even_if_model_asks_again(
            self, generator, monkeypatch, real_tool_defs, tool_manager):
        """After 2 rounds a final request forbids tools; a stray tool_call in its reply is
        ignored (never executed)."""
        llm = install(monkeypatch,
                      make_response(tool_calls=[OUTLINE_CALL]),
                      make_response(tool_calls=[SEARCH_CALL]),
                      make_response(content="best effort", tool_calls=[SEARCH_CALL]))
        out = generator.generate_response("q", tools=real_tool_defs, tool_manager=tool_manager)
        assert out == "best effort"
        assert tool_manager.execute_tool.call_count == 2
        assert len(llm.calls) == 3 and llm.calls[2]["tool_choice"] == "none"

    def test_empty_final_answer_after_two_rounds_returns_fallback_text(
            self, generator, monkeypatch, real_tool_defs, tool_manager):
        install(monkeypatch,
                make_response(tool_calls=[OUTLINE_CALL]),
                make_response(tool_calls=[SEARCH_CALL]),
                make_response(content=None, finish_reason="length"))
        out = generator.generate_response("q", tools=real_tool_defs, tool_manager=tool_manager)
        assert out == AIGenerator.EMPTY_ANSWER

    def test_parallel_calls_in_one_round_count_as_one_round(
            self, generator, monkeypatch, real_tool_defs, tool_manager):
        first = [make_tool_call("search_course_content", {"query": "a"}, "id_a"),
                 make_tool_call("search_course_content", {"query": "b"}, "id_b")]
        llm = install(monkeypatch,
                      make_response(tool_calls=first),
                      make_response(tool_calls=[SEARCH_CALL]),
                      make_response(content="done"))
        generator.generate_response("q", tools=real_tool_defs, tool_manager=tool_manager)
        assert tool_manager.execute_tool.call_count == 3
        assert [m["tool_call_id"] for m in tool_messages(llm.calls[2])] == ["id_a", "id_b", "call_search"]

    def test_empty_search_result_is_not_a_failure(self, generator, monkeypatch, mock_store):
        """With a real ToolManager, "No relevant content found" is a normal result: the model may
        keep going and make its second call."""
        tm = ToolManager()
        tm.register_tool(CourseSearchTool(mock_store))
        tm.register_tool(CourseOutlineTool(mock_store))
        mock_store.search.return_value = empty_search_results()
        llm = install(monkeypatch,
                      make_response(tool_calls=[make_tool_call("search_course_content", {"query": "zzz"}, "c1")]),
                      make_response(tool_calls=[make_tool_call("search_course_content", {"query": "yyy"}, "c2")]),
                      make_response(content="done"))
        out = generator.generate_response("q", tools=tm.get_tool_definitions(), tool_manager=tm)
        assert out == "done"
        assert [c["tool_choice"] for c in llm.calls] == ["auto", "auto", "none"]


# --------------------------------------------------------------------------------------
# A failed tool call ends the tool loop (and is reported to the model, not raised)
# --------------------------------------------------------------------------------------
class TestToolFailureEndsTheLoop:
    @staticmethod
    def assert_wrapped_up_after_one_round(llm, out):
        assert out == "wrapped up"
        assert len(llm.calls) == 2                       # no further tool round was attempted
        assert llm.calls[1]["tool_choice"] == "none"
        assert llm.calls[1].get("tools")

    def test_exception_from_tool_ends_loop(self, generator, monkeypatch, real_tool_defs, tool_manager):
        tool_manager.execute_tool.side_effect = RuntimeError("db down")
        llm = install(monkeypatch, make_response(tool_calls=[SEARCH_CALL]), make_response(content="wrapped up"))
        out = generator.generate_response("q", tools=real_tool_defs, tool_manager=tool_manager)
        self.assert_wrapped_up_after_one_round(llm, out)
        assert "db down" in tool_messages(llm.calls[1])[0]["content"]

    @pytest.mark.parametrize("bad_args", ["{not json", "[1, 2]"])
    def test_invalid_arguments_end_loop(self, generator, monkeypatch, real_tool_defs, tool_manager, bad_args):
        llm = install(monkeypatch,
                      make_response(tool_calls=[make_tool_call("search_course_content", bad_args)]),
                      make_response(content="wrapped up"))
        out = generator.generate_response("q", tools=real_tool_defs, tool_manager=tool_manager)
        self.assert_wrapped_up_after_one_round(llm, out)
        tool_manager.execute_tool.assert_not_called()

    def test_unknown_tool_reported_by_real_tool_manager_ends_loop(self, generator, monkeypatch, mock_store):
        tm = ToolManager()
        tm.register_tool(CourseSearchTool(mock_store))
        llm = install(monkeypatch,
                      make_response(tool_calls=[make_tool_call("no_such_tool", {}, "c1")]),
                      make_response(content="wrapped up"))
        out = generator.generate_response("q", tools=tm.get_tool_definitions(), tool_manager=tm)
        self.assert_wrapped_up_after_one_round(llm, out)
        assert "not found" in tool_messages(llm.calls[1])[0]["content"]

    def test_failure_in_second_round_ends_loop(self, generator, monkeypatch, real_tool_defs, tool_manager):
        tool_manager.execute_tool.side_effect = ["outline text", RuntimeError("boom")]
        llm = install(monkeypatch,
                      make_response(tool_calls=[OUTLINE_CALL]),
                      make_response(tool_calls=[SEARCH_CALL]),
                      make_response(content="wrapped up"))
        out = generator.generate_response("q", tools=real_tool_defs, tool_manager=tool_manager)
        assert out == "wrapped up"
        assert len(llm.calls) == 3 and llm.calls[2]["tool_choice"] == "none"
        assert "boom" in tool_messages(llm.calls[2])[-1]["content"]

    def test_result_text_starting_with_error_is_not_a_failure(
            self, generator, monkeypatch, real_tool_defs, tool_manager):
        """Only calls that could not run are failures, not whatever text a tool returns."""
        tool_manager.execute_tool.return_value = "Error: this is just document text"
        install(monkeypatch,
                make_response(tool_calls=[SEARCH_CALL]),
                make_response(tool_calls=[SEARCH_CALL]),
                make_response(content="done"))
        generator.generate_response("q", tools=real_tool_defs, tool_manager=tool_manager)
        assert tool_manager.execute_tool.call_count == 2

    def test_failed_call_in_parallel_batch_still_answers_every_tool_call_id(
            self, generator, monkeypatch, real_tool_defs, tool_manager):
        calls = [make_tool_call("search_course_content", {"query": "a"}, "id_a"),
                 make_tool_call("search_course_content", "{bad", "id_b"),
                 make_tool_call("search_course_content", {"query": "c"}, "id_c")]
        llm = install(monkeypatch, make_response(tool_calls=calls), make_response(content="wrapped up"))
        generator.generate_response("q", tools=real_tool_defs, tool_manager=tool_manager)
        assert [m["tool_call_id"] for m in tool_messages(llm.calls[1])] == ["id_a", "id_b", "id_c"]
        assert tool_manager.execute_tool.call_count == 2

    def test_api_error_propagates(self, generator, monkeypatch, real_tool_defs, tool_manager):
        install(monkeypatch, make_response(tool_calls=[SEARCH_CALL]), ConnectionError("provider down"))
        with pytest.raises(ConnectionError):
            generator.generate_response("q", tools=real_tool_defs, tool_manager=tool_manager)
