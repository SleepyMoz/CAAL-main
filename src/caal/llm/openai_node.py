"""Simplified OpenAI LLM Node.

This module provides a custom llm_node implementation that bypasses LiveKit's
default LLM wrapper to enable direct OpenAI compatible calls with MCP support.
"""

import inspect
import json
import logging
import time
from collections.abc import AsyncIterable
from typing import Any, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageToolCall

from ..integrations.n8n import execute_n8n_workflow
from ..utils.formatting import strip_markdown_for_tts

logger = logging.getLogger(__name__)


class ToolDataCache:
    """Caches recent tool response data for context injection."""

    def __init__(self, max_entries: int = 3):
        self.max_entries = max_entries
        self._cache: list[dict] = []

    def add(self, tool_name: str, data: Any) -> None:
        """Add tool response data to cache."""
        entry = {"tool": tool_name, "data": data, "timestamp": time.time()}
        self._cache.append(entry)
        if len(self._cache) > self.max_entries:
            self._cache.pop(0)  # Remove oldest

    def get_context_message(self) -> str | None:
        """Format cached data as context string for LLM injection."""
        if not self._cache:
            return None
        parts = ["Recent tool response data for reference:"]
        for entry in self._cache:
            parts.append(f"\n{entry['tool']}: {json.dumps(entry['data'])}")
        return "\n".join(parts)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()


async def openai_llm_node(
    agent,
    chat_ctx,
    model: str = "gpt-4o",
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int | None = None,
    tool_data_cache: ToolDataCache | None = None,
    max_turns: int = 20,
) -> AsyncIterable[str]:
    """Custom LLM node using OpenAI compatible API.

    Args:
        agent: The Agent instance (self)
        chat_ctx: Chat context from LiveKit
        model: Model name
        api_key: OpenAI API key
        base_url: OpenAI base URL
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        max_tokens: Max tokens to generate
        tool_data_cache: Cache for structured tool response data
        max_turns: Max conversation turns to keep in sliding window

    Yields:
        String chunks for TTS output
    """
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    extra_body = {}
    # Some models/providers might support "think" or similar params, 
    # but strictly speaking standard OpenAI doesn't. 
    # If the user was using Qwen via Ollama before, they might miss "think",
    # but we are switching to generic OpenAI compatible, so we remove specific "think" param support
    # unless passed via extra_body if needed. For now we keep it standard.

    try:
        # Build messages from chat context with sliding window
        messages = _build_messages_from_context(
            chat_ctx,
            tool_data_cache=tool_data_cache,
            max_turns=max_turns,
        )

        # Discover tools from agent and MCP servers
        openai_tools = await _discover_tools(agent)

        # If tools available, check for tool calls first (non-streaming)
        if openai_tools:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=openai_tools,
                stream=False,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

            choice = response.choices[0]
            if choice.message.tool_calls:
                tool_calls = choice.message.tool_calls
                logger.info(f"OpenAI returned {len(tool_calls)} tool call(s)")

                # Track tool usage for frontend indicator
                tool_names = [tc.function.name for tc in tool_calls]
                tool_params = []
                for tc in tool_calls:
                    try:
                        tool_params.append(json.loads(tc.function.arguments))
                    except json.JSONDecodeError:
                         tool_params.append({})

                # Publish tool status immediately
                if hasattr(agent, "_on_tool_status") and agent._on_tool_status:
                    import asyncio
                    asyncio.create_task(agent._on_tool_status(True, tool_names, tool_params))

                # Execute tools and get results
                messages = await _execute_tool_calls(
                    agent, messages, list(tool_calls), choice.message,
                    tool_data_cache=tool_data_cache,
                )

                # Stream follow-up response with tool results
                # Note: creating a new client/request for follow up
                response_stream = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )

                async for chunk in response_stream:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        yield strip_markdown_for_tts(delta.content)
                return

            # No tool calls - return content directly
            elif choice.message.content:
                # Publish no-tool status immediately
                if hasattr(agent, "_on_tool_status") and agent._on_tool_status:
                    import asyncio
                    asyncio.create_task(agent._on_tool_status(False, [], []))
                yield strip_markdown_for_tts(choice.message.content)
                return

        # No tools or no tool calls - stream directly
        if hasattr(agent, "_on_tool_status") and agent._on_tool_status:
            import asyncio
            asyncio.create_task(agent._on_tool_status(False, [], []))

        response_stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=None, # Explicitly no tools in this path if we didn't discover any
            stream=True,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        async for chunk in response_stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield strip_markdown_for_tts(delta.content)

    except Exception as e:
        logger.error(f"Error in openai_llm_node: {e}", exc_info=True)
        yield f"I encountered an error: {e}"


def _build_messages_from_context(
    chat_ctx,
    tool_data_cache: ToolDataCache | None = None,
    max_turns: int = 20,
) -> list[dict]:
    """Build OpenAI messages with sliding window and tool data context."""
    system_prompt = None
    chat_messages = []

    for item in chat_ctx.items:
        item_type = type(item).__name__

        if item_type == "ChatMessage":
            msg = {"role": item.role, "content": item.text_content}
            if item.role == "system":
                system_prompt = msg
            else:
                chat_messages.append(msg)
        elif item_type == "FunctionCall":
            # Map LiveKit function call to ephemeral message format if needed
            # But standard LiveKit chat context usually maintains history well.
            # Here we follow ollama_node's approach of reconstructing context
            pass
        # Note: We might need to be careful about preserving tool call history if 
        # LiveKit's ChatContext doesn't store it in a way that maps 1:1 to OpenAI's expectations
        # for "messages" array.
        # Ollama_node implementation reconstructed it carefully.
        # simplified here to just text messages + system prompt + tool context
        # If we need multi-turn tool history, we need to inspect chat_ctx item structure more closely.
        # The ollama_node implementation handled FunctionCall and FunctionCallOutput.
        
    # Re-using logic from ollama_node but adapting for OpenAI format
    # OpenAI expects: tool_calls in "assistant" message, and tool_call_id in "tool" message.
    
    chat_messages = []
    
    for item in chat_ctx.items:
        item_type = type(item).__name__
        
        if item_type == "ChatMessage":
            msg = {"role": item.role, "content": item.text_content}
            if item.role == "system":
                system_prompt = msg
            else:
                chat_messages.append(msg)
        elif item_type == "FunctionCall":
             # We need to append this as an assistant message with tool_calls
             pass 
             # LiveKit context history handling can be tricky. 
             # For now, let's assume we primarily rely on the current turn's context 
             # processing in `openai_llm_node` loop for tool execution within a turn.
             # Historical tool calls in previous turns might be stored as text or specialized items.
             
             # Re-implementing from ollama_node to be safe:
             try:
                chat_messages.append({
                    "role": "assistant",
                    "content": None, # OpenAI allows null content if tool_calls present
                    "tool_calls": [{
                        "id": item.id,
                        "type": "function",
                        "function": {
                            "name": item.name,
                            "arguments": getattr(item, "arguments", {}) or {},
                        },
                    }],
                })
             except AttributeError:
                pass
        elif item_type == "FunctionCallOutput":
            try:
                chat_messages.append({
                    "role": "tool",
                    "content": str(item.content),
                    "tool_call_id": item.tool_call_id,
                })
            except AttributeError:
                pass

    # Build final message list
    messages = []

    # 1. System prompt always first
    if system_prompt:
        messages.append(system_prompt)

    # 2. Inject tool data context
    if tool_data_cache:
        context = tool_data_cache.get_context_message()
        if context:
            messages.append({"role": "system", "content": context})

    # 3. Apply sliding window
    max_messages = max_turns * 2
    if len(chat_messages) > max_messages:
        chat_messages = chat_messages[-max_messages:]

    messages.extend(chat_messages)
    return messages


async def _discover_tools(agent) -> list[dict] | None:
    """Discover tools and return in OpenAI format."""
    # Use existing cache
    if hasattr(agent, "_openai_tools_cache") and agent._openai_tools_cache is not None:
        return agent._openai_tools_cache

    tools = []

    # Agent methods
    if hasattr(agent, "_tools") and agent._tools:
        for tool in agent._tools:
            if hasattr(tool, "__func__"):
                func = tool.__func__
                name = func.__name__
                description = func.__doc__ or ""
                sig = inspect.signature(func)
                properties = {}
                required = []

                for param_name, param in sig.parameters.items():
                    if param_name == "self":
                        continue
                    param_type = "string"
                    if param.annotation != inspect.Parameter.empty:
                        if param.annotation is str:
                           param_type = "string"
                        elif param.annotation is int:
                            param_type = "integer"
                        elif param.annotation is float:
                            param_type = "number"
                        elif param.annotation is bool:
                            param_type = "boolean"
                    properties[param_name] = {"type": param_type}
                    if param.default == inspect.Parameter.empty and param_name != "self":
                        required.append(param_name)

                tools.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    },
                })

    # MCP tools
    if hasattr(agent, "_caal_mcp_servers") and agent._caal_mcp_servers:
        for server_name, server in agent._caal_mcp_servers.items():
            if server_name == "n8n":
                continue
            
            mcp_tools = await _get_mcp_tools(server)
            for tool in mcp_tools:
                original_name = tool["function"]["name"]
                tool["function"]["name"] = f"{server_name}__{original_name}"
            tools.extend(mcp_tools)

    # n8n tools
    if hasattr(agent, "_n8n_workflow_tools") and agent._n8n_workflow_tools:
        tools.extend(agent._n8n_workflow_tools)

    result = tools if tools else None
    agent._openai_tools_cache = result
    return result


async def _get_mcp_tools(mcp_server) -> list[dict]:
    """Get tools from an MCP server in OpenAI format."""
    tools = []
    if not mcp_server or not hasattr(mcp_server, "_client") or not mcp_server._client:
        return tools

    try:
        tools_result = await mcp_server._client.list_tools()
        if hasattr(tools_result, "tools"):
            for mcp_tool in tools_result.tools:
                parameters = {"type": "object", "properties": {}, "required": []}
                if hasattr(mcp_tool, "inputSchema") and mcp_tool.inputSchema:
                    schema = mcp_tool.inputSchema
                    if isinstance(schema, dict):
                        parameters = schema
                    elif hasattr(schema, "properties"):
                        parameters["properties"] = schema.properties or {}
                        parameters["required"] = getattr(schema, "required", []) or []

                tools.append({
                    "type": "function",
                    "function": {
                        "name": mcp_tool.name,
                        "description": getattr(mcp_tool, "description", "") or "",
                        "parameters": parameters,
                    },
                })
    except Exception as e:
        logger.warning(f"Error getting MCP tools: {e}")

    return tools


async def _execute_tool_calls(
    agent,
    messages: list[dict],
    tool_calls: list[ChatCompletionMessageToolCall],
    response_message: Any,
    tool_data_cache: ToolDataCache | None = None,
) -> list[dict]:
    """Execute tool calls and append results to messages."""
    
    # Add assistant message with tool calls
    # For OpenAI API, we must pass the exact tool calls object structure or dictionary equivalent
    
    assistant_msg = {
        "role": "assistant",
        "content": response_message.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            } for tc in tool_calls
        ]
    }
    
    messages.append(assistant_msg)

    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments_str = tool_call.function.arguments or "{}"
        try:
             arguments = json.loads(arguments_str)
        except:
             arguments = {}
             
        logger.info(f"Executing tool: {tool_name} with args: {arguments}")

        try:
            tool_result = await _execute_single_tool(agent, tool_name, arguments)

            if tool_data_cache and isinstance(tool_result, dict):
                data = tool_result.get("data") or tool_result.get("results") or tool_result
                tool_data_cache.add(tool_name, data)

            messages.append({
                "role": "tool",
                "content": str(tool_result),
                "tool_call_id": tool_call.id,
            })
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {e}"
            logger.error(error_msg, exc_info=True)
            messages.append({
                "role": "tool",
                "content": error_msg,
                "tool_call_id": tool_call.id,
            })

    return messages


async def _execute_single_tool(agent, tool_name: str, arguments: dict) -> Any:
    # Same routing logic as before
    if hasattr(agent, tool_name) and callable(getattr(agent, tool_name)):
        return await getattr(agent, tool_name)(**arguments)

    if (
        hasattr(agent, "_n8n_workflow_name_map")
        and tool_name in agent._n8n_workflow_name_map
        and hasattr(agent, "_n8n_base_url")
        and agent._n8n_base_url
    ):
        workflow_name = agent._n8n_workflow_name_map[tool_name]
        return await execute_n8n_workflow(agent._n8n_base_url, workflow_name, arguments)

    if hasattr(agent, "_caal_mcp_servers") and agent._caal_mcp_servers:
        if "__" in tool_name:
            server_name, actual_tool = tool_name.split("__", 1)
        else:
            server_name, actual_tool = "n8n", tool_name

        if server_name in agent._caal_mcp_servers:
            server = agent._caal_mcp_servers[server_name]
            # _call_mcp_tool logic is same, will duplicate here for completeness 
            # or could import - but imports might be circular. I'll inline it.
            return await _call_mcp_tool(server, actual_tool, arguments)

    raise ValueError(f"Tool {tool_name} not found")

async def _call_mcp_tool(mcp_server, tool_name: str, arguments: dict) -> Any | None:
    if not mcp_server or not hasattr(mcp_server, "_client"):
        return None
    try:
        result = await mcp_server._client.call_tool(tool_name, arguments)
        if result.isError:
             return f"MCP tool error: {result.content}"
        
        text_contents = []
        for content in result.content:
            if hasattr(content, "text") and content.text:
                text_contents.append(content.text)
        return "\n".join(text_contents) if text_contents else "Tool executed successfully"
    except Exception as e:
        logger.warning(f"Error calling MCP tool {tool_name}: {e}")
        return None
