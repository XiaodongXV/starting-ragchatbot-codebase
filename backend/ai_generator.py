import json
import os
from typing import List, Optional, Dict, Any, Tuple

# litellm downloads a tiktoken encoding at import time, which fails behind
# Zscaler's self-signed certificate unless truststore is active first.
import truststore
truststore.inject_into_ssl()

import litellm

from config import PROVIDERS
from search_tools import ToolError

class AIGenerator:
    """Handles interactions with an LLM (via LiteLLM) for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Tool Usage:
- **Outline questions** (structure, syllabus, lesson list, "what lessons does X have"): use `get_course_outline` — do NOT use the content search tool for these
- **Content questions** (explanations, concepts, details from lesson material): use `search_course_content`
- **Up to 2 sequential tool calls per query.** You may call a tool, read its result, then make a second call that depends on it (e.g. get a lesson title with `get_course_outline`, then search for that topic with `search_course_content`). Make a second call only when the answer depends on the first result; otherwise answer right away
- Do not repeat a call that already returned what you need; if a tool returns an error or nothing useful, stop calling tools and answer from what you have
- **Finding courses on a topic**: search WITHOUT `course_name` so every course is covered. When the topic came from a known course (e.g. a lesson of course X), also set `exclude_course` to that course so only OTHER courses are returned. Use `course_name` / `lesson_number` only when the user named that course or lesson. A lesson title from `get_course_outline` already describes its topic: use it directly as the query of that one search — do not first search inside the source course to read the lesson
- **Only state what the tool results contain.** Name only courses, lessons and facts that appear in them — never add courses or details from your own knowledge. If the results do not answer the question (or a tool returns nothing), say plainly that the course materials have nothing on it
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    # Max API rounds in which the model may call tools; a final text-only request follows.
    MAX_TOOL_ROUNDS = 2

    def __init__(self, api_key: str, model: str, fallback_models: Optional[List[str]] = None):
        self.model = model

        # LiteLLM reads credentials from the environment, keyed by provider.
        # Only the active provider's key needs to be present.
        provider = model.split("/", 1)[0]
        if api_key and provider in PROVIDERS:
            os.environ.setdefault(PROVIDERS[provider][1], api_key)

        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
        # LiteLLM tries these in order if the primary model raises
        # (quota exhaustion, overload, ...). Applies to both calls below.
        if fallback_models:
            self.base_params["fallbacks"] = list(fallback_models)

    @staticmethod
    def _to_openai_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Translate Anthropic-shaped tool definitions into OpenAI format"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            for tool in tools
        ]

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]
        openai_tools = self._to_openai_tools(tools) if tools else None

        # Each round is its own API request, so the model sees earlier tool results before
        # deciding whether to call another tool. Rounds stop when the model answers in text,
        # when MAX_TOOL_ROUNDS is reached, or when a tool call fails.
        for _ in range(self.MAX_TOOL_ROUNDS):
            params = {**self.base_params, "messages": messages}
            if openai_tools:
                params["tools"] = openai_tools
                params["tool_choice"] = "auto"
            message = litellm.completion(**params).choices[0].message

            if not (message.tool_calls and tool_manager):
                return self._answer_text(message.content)

            # Keep the provider's own assistant object: it may carry provider-specific fields
            messages.append(message)
            failed = False
            for tool_call in message.tool_calls:
                result, ok = self._run_tool(tool_manager, tool_call)
                failed = failed or not ok
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
            if failed:
                break

        # Final answer after the last round (limit reached or a tool failed). The history now
        # contains tool_calls/tool messages, which stricter providers only accept when tools are
        # declared, so declare them but forbid further calls.
        final = {**self.base_params, "messages": messages}
        if openai_tools:
            final["tools"] = openai_tools
            final["tool_choice"] = "none"
        return self._answer_text(litellm.completion(**final).choices[0].message.content)

    # Shown when the model returns no text (e.g. reasoning used up max_tokens). The API response
    # model requires a str, so None would otherwise turn into HTTP 500.
    EMPTY_ANSWER = "Sorry, I couldn't generate an answer this time. Please try again."

    @classmethod
    def _answer_text(cls, content: Optional[str]) -> str:
        return content if content and content.strip() else cls.EMPTY_ANSWER

    @staticmethod
    def _run_tool(tool_manager, tool_call) -> Tuple[str, bool]:
        """Execute one model-requested tool call and return (text, ok). Any problem is returned
        as text for the model to read instead of raising; ok=False marks a call that could not
        run (a normal result such as "No relevant content found" is still ok)."""
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except (json.JSONDecodeError, TypeError) as e:
            return f"Error: arguments for tool '{name}' are not valid JSON ({e})", False
        if not isinstance(args, dict):
            return f"Error: arguments for tool '{name}' must be a JSON object", False
        try:
            result = tool_manager.execute_tool(name, **args)
        except Exception as e:
            return f"Error: tool '{name}' failed: {e}", False
        return result, not isinstance(result, ToolError)
