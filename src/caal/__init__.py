"""
CAAL - Voice Assistant
======================

A modular voice assistant with n8n workflow integrations and OpenAI-compatible LLM support.

Core Components:
    OpenAILLM: OpenAI-compatible LLM integration

Integrations:
    n8n: Workflow discovery and execution via n8n MCP

Example:
    >>> from caal import OpenAILLM
    >>> from caal.integrations import load_mcp_config
    >>>
    >>> llm = OpenAILLM()
    >>> mcp_configs = load_mcp_config()

Repository: https://github.com/CoreWorxLab/caal
License: MIT
"""

__version__ = "0.1.0"
__author__ = "CoreWorxLab"

from .llm import OpenAILLM

__all__ = [
    "OpenAILLM",
    "__version__",
]
