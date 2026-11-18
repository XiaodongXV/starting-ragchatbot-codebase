"""Layer-by-layer reproduction against the REAL provider and a COPY of the real index.

Opt-in (sends a handful of requests to the configured LLM provider):
    cd backend && uv run pytest tests/test_live_diagnostics.py -m live -s -v

The FIRST red layer is where the content-question pipeline breaks:
    L0 provider reachable (no tools)       -> key / model / network problem, not our code
    L1 vector search on the real index     -> Chroma / embeddings / data
    L2 CourseSearchTool.execute            -> our tool
    L3 first LLM call asks for the tool    -> prompt / tool schema / model behaviour
    L4 follow-up LLM call with tool result -> H1 / H2 (replayed tool_calls, empty content)
    L5 RAGSystem.query                     -> whole orchestration
"""
import copy
import dataclasses
import shutil

import pytest

pytestmark = pytest.mark.live

import litellm  # noqa: E402  (conftest has already injected truststore)

from ai_generator import AIGenerator  # noqa: E402
from config import config  # noqa: E402
from conftest import BACKEND_DIR, EMBEDDING_MODEL  # noqa: E402
from rag_system import RAGSystem  # noqa: E402
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager  # noqa: E402
from vector_store import VectorStore  # noqa: E402

QUESTION = "According to the Computer Use course, how does the model control a computer?"

if not config.LLM_API_KEY:
    pytest.skip(f"no API key for provider {config.LLM_PROVIDER!r}", allow_module_level=True)


def _describe(resp):
    choice = resp.choices[0]
    msg = choice.message
    return (f"model={getattr(resp, 'model', '?')} finish_reason={choice.finish_reason} "
            f"content={msg.content!r} tool_calls="
            f"{[(t.function.name, t.function.arguments) for t in (msg.tool_calls or [])]} "
            f"usage={getattr(resp, 'usage', None)}")


def _call(label, **kwargs):
    try:
        return litellm.completion(**kwargs)
    except Exception as e:  # show the provider's own words
        pytest.fail(f"{label} raised {type(e).__name__}: {e}", pytrace=False)


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    src = BACKEND_DIR / "chroma_db"
    if not src.exists():
        pytest.skip("backend/chroma_db missing")
    dst = tmp_path_factory.mktemp("live_db") / "copy"
    shutil.copytree(src, dst)
    return str(dst)


@pytest.fixture(scope="module")
def store(db_path):
    return VectorStore(db_path, EMBEDDING_MODEL, config.MAX_RESULTS)


@pytest.fixture(scope="module")
def tool_manager(store):
    tm = ToolManager()
    tm.register_tool(CourseSearchTool(store))
    tm.register_tool(CourseOutlineTool(store))
    return tm


@pytest.fixture(scope="module")
def generator():
    return AIGenerator(config.LLM_API_KEY, config.LLM_MODEL, config.LLM_FALLBACK_MODELS)


@pytest.fixture(scope="module")
def api_params(generator, tool_manager):
    """Exactly what generate_response() sends as its first call."""
    return {
        **generator.base_params,
        "messages": [{"role": "system", "content": AIGenerator.SYSTEM_PROMPT},
                     {"role": "user", "content": f"Answer this question about course materials: {QUESTION}"}],
        "tools": generator._to_openai_tools(tool_manager.get_tool_definitions()),
        "tool_choice": "auto",
    }


@pytest.fixture(scope="module")
def first_response(api_params):
    resp = _call("L3 first LLM call (with tools)", **api_params)
    print("\n[L3]", _describe(resp))
    return resp


# ---------------------------------------------------------------- L0
def test_L0_provider_answers_a_plain_question(generator):
    print(f"\n[config] provider={config.LLM_PROVIDER} model={config.LLM_MODEL} fallbacks={config.LLM_FALLBACK_MODELS}")
    out = generator.generate_response("What is 2+2? Answer with just the number.")
    print("[L0] ->", repr(out))
    assert isinstance(out, str) and "4" in out


# ---------------------------------------------------------------- L1 / L2
def test_L1_vector_search_on_real_index(store):
    res = store.search("computer use mouse clicks keystrokes")
    assert res.error is None, res.error
    assert not res.is_empty()
    print("\n[L1]", [(m["course_title"], m["lesson_number"]) for m in res.metadata])


def test_L2_search_tool_on_real_index(tool_manager):
    out = tool_manager.execute_tool("search_course_content", query="how does the model control a computer",
                                    course_name="Computer Use")
    print("\n[L2]", out[:300].replace("\n", " "))
    assert out.startswith("[") and "Search error" not in out


# ---------------------------------------------------------------- L3
def test_L3_model_requests_the_search_tool(first_response):
    msg = first_response.choices[0].message
    assert msg.tool_calls, f"model answered without using a tool: {_describe(first_response)}"
    assert msg.tool_calls[0].function.name in {"search_course_content", "get_course_outline"}


# ---------------------------------------------------------------- L4
@pytest.fixture(scope="module")
def first_message(first_response):
    """Pristine copy of the model's tool-call message. litellm mutates message objects it is given
    (tool_calls become dicts), so each follow-up must work on its own deep copy."""
    return copy.deepcopy(first_response.choices[0].message)


def _followup(generator, first_message, api_params, tool_manager, with_tools):
    """Replicates AIGenerator._handle_tool_execution, optionally declaring tools on the 2nd call."""
    import json
    msg = copy.deepcopy(first_message)
    messages = list(api_params["messages"]) + [msg]
    for tc in msg.tool_calls:
        result = tool_manager.execute_tool(tc.function.name, **json.loads(tc.function.arguments))
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    params = {**generator.base_params, "messages": messages}
    if with_tools:
        params["tools"] = api_params["tools"]
    return _call(f"L4 follow-up call (with_tools={with_tools})", **params)


def test_L4a_followup_without_tools(generator, first_message, api_params, tool_manager):
    """Second call WITHOUT tools — what the app did before the follow-up started declaring tools."""
    resp = _followup(generator, first_message, api_params, tool_manager, with_tools=False)
    print("\n[L4a]", _describe(resp))
    assert resp.choices[0].message.content, f"empty answer: {_describe(resp)}"


def test_L4b_followup_with_tools_declared(generator, first_message, api_params, tool_manager):
    """Same, but declaring tools. If 4a is red and 4b green -> H1 confirmed."""
    resp = _followup(generator, first_message, api_params, tool_manager, with_tools=True)
    print("\n[L4b]", _describe(resp))
    assert resp.choices[0].message.content or resp.choices[0].message.tool_calls, _describe(resp)


def test_L4c_generate_response_end_to_end(generator, tool_manager):
    out = generator.generate_response(
        f"Answer this question about course materials: {QUESTION}",
        tools=tool_manager.get_tool_definitions(), tool_manager=tool_manager)
    print("\n[L4c]", repr(out))
    assert isinstance(out, str) and out.strip()


# ---------------------------------------------------------------- L5
def test_L5_rag_system_query(db_path):
    rag = RAGSystem(dataclasses.replace(config, CHROMA_PATH=db_path))
    answer, sources = rag.query(QUESTION)
    print("\n[L5] answer:", answer, "\n[L5] sources:", sources)
    assert isinstance(answer, str) and answer.strip()
    assert sources, "content question produced no sources"
