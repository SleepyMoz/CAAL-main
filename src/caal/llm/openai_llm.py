"""
OpenAI LLM Plugin for LiveKit Agents
====================================

OpenAI compatible LLM integration.

This plugin provides an LLM interface that satisfies LiveKit's requirements
while allowing the VoiceAssistant's llm_node override to handle actual
LLM calls with MCP tool integration.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from livekit.agents import llm
from livekit.agents.llm import ChatChunk, ChatContext, ChoiceDelta
from livekit.agents.llm.tool_context import FunctionTool, RawFunctionTool
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

__all__ = ["OpenAILLM"]

logger = logging.getLogger(__name__)


class OpenAILLM(llm.LLM):
    """
    LiveKit LLM plugin for OpenAI compatible APIs.

    This plugin is designed to be used with a VoiceAssistant that overrides
    the llm_node method. The actual LLM calls are handled by openai_llm_node(),
    which supports MCP tool discovery and execution.

    Args:
        model: Model name (e.g., "gpt-4o", "llama-3.1-70b")
        api_key: OpenAI API key
        base_url: OpenAI implementation base URL
        temperature: Sampling temperature (0.0-2.0)
        top_p: Nucleus sampling threshold (0.0-1.0)
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens

        logger.debug(f"OpenAILLM initialized: {model}")

    # === Required LLM interface properties ===

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "openai"

    # === Configuration accessors for llm_node ===

    @property
    def api_key(self) -> str | None:
        return self._api_key

    @property
    def base_url(self) -> str | None:
        return self._base_url

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def top_p(self) -> float:
        return self._top_p

    @property
    def max_tokens(self) -> int | None:
        return self._max_tokens

    # === Required LLM interface method ===

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> llm.LLMStream:
        return _OpenAILLMStream(
            llm=self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )


class _OpenAILLMStream(llm.LLMStream):
    """
    Minimal LLMStream implementation for interface compliance.
    """

    def __init__(
        self,
        llm: OpenAILLM,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool | RawFunctionTool],
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)

    async def _run(self) -> None:
        # Emit a minimal response for interface compliance
        request_id = str(uuid.uuid4())
        logger.warning(
            "OpenAILLM._run() called directly - this usually means llm_node "
            "override is not active. Using fallback response."
        )

        chunk = ChatChunk(
            id=request_id,
            delta=ChoiceDelta(
                role="assistant",
                content="I'm configured to use a custom LLM node. "
                        "Please ensure the agent's llm_node override is active.",
            ),
        )
        self._event_ch.send_nowait(chunk)
