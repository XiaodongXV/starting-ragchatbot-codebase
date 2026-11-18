"""Helpers that fake litellm responses (OpenAI chat-completion shape)."""
import json
from types import SimpleNamespace as NS


def make_tool_call(name, args, call_id="call_1"):
    """Build an OpenAI-style tool call. `args` may be a dict or a raw string."""
    raw = args if isinstance(args, str) else json.dumps(args)
    return NS(id=call_id, type="function", function=NS(name=name, arguments=raw))


def make_response(content=None, tool_calls=None, finish_reason=None):
    """Build a litellm-like completion response."""
    message = NS(role="assistant", content=content, tool_calls=tool_calls or None)
    reason = finish_reason or ("tool_calls" if tool_calls else "stop")
    return NS(choices=[NS(message=message, finish_reason=reason)])


class ScriptedLLM:
    """Drop-in for `litellm.completion`: returns queued responses in order,
    raises queued exceptions, and records a snapshot of every call."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, **kwargs):
        snapshot = dict(kwargs)
        if "messages" in snapshot:
            snapshot["messages"] = list(snapshot["messages"])
        self.calls.append(snapshot)
        if not self.responses:
            raise AssertionError("ScriptedLLM: more LLM calls than scripted responses")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
