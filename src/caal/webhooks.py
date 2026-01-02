"""Webhook server for external triggers (announcements, tool reload, wake word, settings).

This module provides HTTP endpoints that allow external systems (like n8n)
and the frontend to trigger actions on the running voice agent.

Endpoints:
    POST /announce      - Make the agent speak a message
    POST /reload-tools  - Refresh MCP tool cache and optionally announce
    POST /wake          - Handle wake word detection (greet user)
    GET  /health        - Health check
    GET  /settings      - Get current settings
    POST /settings      - Update settings
    GET  /prompt        - Get current prompt content
    POST /prompt        - Save custom prompt
    GET  /voices        - List available TTS voices
    GET  /models        - List available LLM models

Usage:
    # Start in a background thread from voice_agent.py:
    import threading
    import uvicorn
    from caal.webhooks import app

    def run_webhook_server():
        uvicorn.run(app, host="0.0.0.0", port=8889, log_level="info")

    webhook_thread = threading.Thread(target=run_webhook_server, daemon=True)
    webhook_thread.start()
"""

from __future__ import annotations

import logging
import os
import random

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import settings as settings_module

logger = logging.getLogger(__name__)

app = FastAPI(
    title="CAAL Webhook API",
    description="External triggers for CAAL voice agent",
    version="1.0.0",
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Frontend can be on different port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnnounceRequest(BaseModel):
    """Request body for /announce endpoint."""

    message: str
    room_name: str = "voice_assistant_room"


class ReloadToolsRequest(BaseModel):
    """Request body for /reload-tools endpoint."""

    tool_name: str | None = None  # Optional: announce specific tool name
    message: str | None = None  # Optional: custom announcement message (overrides tool_name)
    room_name: str = "voice_assistant_room"


class WakeRequest(BaseModel):
    """Request body for /wake endpoint."""

    room_name: str = "voice_assistant_room"


class WakeResponse(BaseModel):
    """Response body for /wake endpoint."""

    status: str
    room_name: str


class AnnounceResponse(BaseModel):
    """Response body for /announce endpoint."""

    status: str
    room_name: str


class ReloadToolsResponse(BaseModel):
    """Response body for /reload-tools endpoint."""

    status: str
    tool_count: int
    room_name: str


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""

    status: str
    active_sessions: list[str]


@app.post("/announce", response_model=AnnounceResponse)
async def announce(req: AnnounceRequest) -> AnnounceResponse:
    """Make the agent speak a message."""
    from . import session_registry

    result = session_registry.get(req.room_name)
    if not result:
        logger.warning(f"Announce failed: no session in room {req.room_name}")
        raise HTTPException(
            status_code=404,
            detail=f"No active session in room: {req.room_name}",
        )

    session, _agent = result
    logger.info(f"Announcing to room {req.room_name}: {req.message[:50]}...")

    # Say the message directly (bypasses LLM for instant response)
    await session.say(req.message)

    return AnnounceResponse(status="announced", room_name=req.room_name)


@app.post("/reload-tools", response_model=ReloadToolsResponse)
async def reload_tools(req: ReloadToolsRequest) -> ReloadToolsResponse:
    """Refresh MCP tool cache and optionally announce new tool availability."""
    from . import session_registry
    from .integrations import n8n

    result = session_registry.get(req.room_name)
    if not result:
        logger.warning(f"Reload failed: no session in room {req.room_name}")
        raise HTTPException(
            status_code=404,
            detail=f"No active session in room: {req.room_name}",
        )

    session, agent = result
    logger.info(f"Reloading tools for room {req.room_name}")

    # Clear all caches
    if hasattr(agent, "_openai_tools_cache"):
        agent._openai_tools_cache = None
    elif hasattr(agent, "_ollama_tools_cache"): # Backward compatibility/cleanup
        agent._ollama_tools_cache = None
        
    n8n.clear_caches()

    # Re-discover n8n workflows if MCP is configured
    tool_count = 0
    if agent._n8n_mcp and agent._n8n_base_url:
        try:
            tools, name_map = await n8n.discover_n8n_workflows(
                agent._n8n_mcp, agent._n8n_base_url
            )
            agent._n8n_workflow_tools = tools
            agent._n8n_workflow_name_map = name_map
            tool_count = len(tools)
            logger.info(f"Discovered {tool_count} n8n workflows")
        except Exception as e:
            logger.error(f"Failed to re-discover n8n workflows: {e}")

    # Announce: custom message takes priority, then tool_name format
    if req.message:
        await session.say(req.message)
    elif req.tool_name:
        await session.say(f"A new tool called '{req.tool_name}' is now available.")

    return ReloadToolsResponse(
        status="reloaded",
        tool_count=tool_count,
        room_name=req.room_name,
    )


@app.post("/wake", response_model=WakeResponse)
async def wake(req: WakeRequest) -> WakeResponse:
    """Handle wake word detection - greet the user."""
    from . import session_registry

    result = session_registry.get(req.room_name)
    if not result:
        logger.warning(f"Wake failed: no session in room {req.room_name}")
        raise HTTPException(
            status_code=404,
            detail=f"No active session in room: {req.room_name}",
        )

    session, _agent = result
    logger.info(f"Wake word detected in room {req.room_name}")

    # Say a random greeting directly (bypasses LLM for instant response)
    greetings = settings_module.get_setting("wake_greetings")
    greeting = random.choice(greetings)
    await session.say(greeting)

    return WakeResponse(status="greeted", room_name=req.room_name)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    from . import session_registry

    return HealthResponse(
        status="ok",
        active_sessions=session_registry.list_rooms(),
    )


# =============================================================================
# Settings Endpoints
# =============================================================================


class SettingsResponse(BaseModel):
    """Response body for /settings endpoint."""

    settings: dict
    prompt_content: str
    custom_prompt_exists: bool


class SettingsUpdateRequest(BaseModel):
    """Request body for POST /settings endpoint."""

    settings: dict


class PromptResponse(BaseModel):
    """Response body for /prompt endpoint."""

    prompt: str  # "default" or "custom"
    content: str
    is_custom: bool


class PromptUpdateRequest(BaseModel):
    """Request body for POST /prompt endpoint."""

    content: str


class VoicesResponse(BaseModel):
    """Response body for /voices endpoint."""

    voices: list[str]


class ModelsResponse(BaseModel):
    """Response body for /models endpoint."""

    models: list[str]


@app.get("/settings", response_model=SettingsResponse)
async def get_settings() -> SettingsResponse:
    """Get current settings and prompt content."""
    settings = settings_module.load_settings()
    prompt_content = settings_module.load_prompt_content()
    custom_exists = settings_module.custom_prompt_exists()

    return SettingsResponse(
        settings=settings,
        prompt_content=prompt_content,
        custom_prompt_exists=custom_exists,
    )


@app.post("/settings", response_model=SettingsResponse)
async def update_settings(req: SettingsUpdateRequest) -> SettingsResponse:
    """Update settings."""
    # Load current settings
    current = settings_module.load_settings()

    # Merge with new settings (only known keys)
    for key, value in req.settings.items():
        if key in settings_module.DEFAULT_SETTINGS:
            current[key] = value

    # Save merged settings
    settings_module.save_settings(current)

    # Reload and return
    settings = settings_module.reload_settings()
    prompt_content = settings_module.load_prompt_content()
    custom_exists = settings_module.custom_prompt_exists()

    logger.info(f"Settings updated: {list(req.settings.keys())}")

    return SettingsResponse(
        settings=settings,
        prompt_content=prompt_content,
        custom_prompt_exists=custom_exists,
    )


@app.get("/prompt", response_model=PromptResponse)
async def get_prompt() -> PromptResponse:
    """Get current prompt content."""
    prompt_name = settings_module.get_setting("prompt", "default")
    content = settings_module.load_prompt_content(prompt_name)
    is_custom = prompt_name == "custom" and settings_module.custom_prompt_exists()

    return PromptResponse(
        prompt=prompt_name,
        content=content,
        is_custom=is_custom,
    )


@app.post("/prompt", response_model=PromptResponse)
async def save_prompt(req: PromptUpdateRequest) -> PromptResponse:
    """Save custom prompt content."""
    # Save to custom.md
    settings_module.save_custom_prompt(req.content)

    # Update settings to use custom prompt
    current = settings_module.load_settings()
    current["prompt"] = "custom"
    settings_module.save_settings(current)

    logger.info("Custom prompt saved and activated")

    return PromptResponse(
        prompt="custom",
        content=req.content,
        is_custom=True,
    )


@app.get("/voices", response_model=VoicesResponse)
async def get_voices() -> VoicesResponse:
    """Get available TTS voices from Kokoro."""
    kokoro_url = os.getenv("KOKORO_URL", "http://kokoro:8880")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{kokoro_url}/v1/audio/voices",
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

            # Kokoro returns {"voices": [{"id": "...", ...}, ...]}
            voices = [v.get("id") or v.get("voice_id") for v in data.get("voices", [])]
            voices = [v for v in voices if v]  # Filter None values

            return VoicesResponse(voices=voices)
    except Exception as e:
        logger.warning(f"Failed to fetch voices from Kokoro: {e}")
        # Return default voices as fallback
        return VoicesResponse(
            voices=["af_heart", "af_bella", "af_sarah", "am_adam", "am_puck"]
        )


@app.get("/models", response_model=ModelsResponse)
async def get_models() -> ModelsResponse:
    """Get available LLM models from OpenAI compatible API."""
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("OPENAI_API_KEY", "")

    # Ensure URL ends with /models, handling potential trailing slashes
    url = f"{base_url.rstrip('/')}/models"
    
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers=headers,
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

            # OpenAI returns {"data": [{"id": "...", ...}, ...]}
            models = [m.get("id") for m in data.get("data", [])]
            models = [m for m in models if m]

            return ModelsResponse(models=models)
    except Exception as e:
        logger.warning(f"Failed to fetch models from OpenAI: {e}")
        # Return fallback or empty
        return ModelsResponse(models=[])
