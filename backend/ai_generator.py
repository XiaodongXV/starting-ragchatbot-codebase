import json
import os
from typing import List, Optional, Dict, Any

# litellm downloads a tiktoken encoding at import time, which fails behind
# Zscaler's self-signed certificate unless truststore is active first.
import truststore
truststore.inject_into_ssl()

import litellm

from config import PROVIDERS

class AIGenerator:
    """Handles interactions with an LLM (via LiteLLM) for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Tool Usage:
- **Outline questions** (structure, syllabus, lesson list, "what lessons does X have"): use `get_course_outline` — do NOT use the content search tool for these
- **Content questions** (explanations, concepts, details from lesson material): use `search_course_content`
- **One tool call per query maximum**
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

    def __init__(self, api_key: str, model: str):
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

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": query},
            ],
        }

        # Add tools if available
        if tools:
            api_params["tools"] = self._to_openai_tools(tools)
            api_params["tool_choice"] = "auto"

        # Get response from the model
        response = litellm.completion(**api_params)
        message = response.choices[0].message

        # Handle tool execution if needed
        if message.tool_calls and tool_manager:
            return self._handle_tool_execution(message, api_params, tool_manager)

        # Return direct response
        return message.content

    def _handle_tool_execution(self, initial_message, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            initial_message: The assistant message containing tool calls
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages plus the assistant's tool call request
        messages = base_params["messages"].copy()
        messages.append(initial_message)

        # Execute each tool call and append its result as its own message
        for tool_call in initial_message.tool_calls:
            tool_result = tool_manager.execute_tool(
                tool_call.function.name,
                **json.loads(tool_call.function.arguments)
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            })

        # Get final response without tools
        final_response = litellm.completion(**{
            **self.base_params,
            "messages": messages,
        })
        return final_response.choices[0].message.content
