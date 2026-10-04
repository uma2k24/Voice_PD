"""Preview or create a dedicated ElevenLabs agent, storing its ID in local .env."""
from __future__ import annotations
import argparse
import asyncio
import json
import os
from pathlib import Path
import httpx
from dotenv import set_key
from src.config import load_config, PROJECT_ROOT
from src.elevenlabs_agent import AGENT_PROMPT, FIRST_MESSAGE

AGENT_NAME = "Voice PD Research Assistant"


def agent_configuration(voice_id="21m00Tcm4TlvDq8ikWAM"):
    return {"name": AGENT_NAME, "conversation_config": {
        "asr": {"user_input_audio_format": "pcm_16000"},
        "tts": {"voice_id": voice_id, "agent_output_audio_format": "pcm_16000"},
        "agent": {"language": "en", "first_message": FIRST_MESSAGE,
                  "prompt": {"prompt": AGENT_PROMPT, "llm": "gemini-3.1-flash-lite", "temperature": .2,
                    "tools": [
                        {"type": "client", "name": "set_recording_task",
                         "description": "Select the actual speech exercise before collecting it. Never label spontaneous speech as a vowel.",
                         "expects_response": True, "response_timeout_secs": 10,
                         "parameters": {"type": "object", "required": ["task"], "properties": {
                             "task": {"type": "string", "description": "The speech task actually being collected.",
                                      "enum": ["conversation", "reading", "sustained_vowel"]}}}},
                        {"type": "client", "name": "get_voice_analysis",
                         "description": "Get the latest completed acoustic measurement and classifier status. Do not guess a result.",
                         "expects_response": True, "response_timeout_secs": 10,
                         "parameters": {"type": "object", "properties": {}}}]}},
        "conversation": {"max_duration_seconds": 600, "client_events": [
            "conversation_initiation_metadata", "ping", "audio", "interruption",
            "user_transcript", "agent_response", "client_tool_call"]}},
        "platform_settings": {"auth": {"enable_auth": True},
            "privacy": {"record_voice": False},
            "overrides": {"conversation_config_override": {"agent": {
                "first_message": True, "prompt": {"prompt": True}}}}}}


def safe_error(response):
    try:
        error = response.json().get("detail", {})
        message = error.get("message", "") if isinstance(error, dict) else str(error)
    except ValueError:
        message = "Provider rejected the request."
    for variable in ("ELEVENLABS_API_KEY", "GEMINI_API_KEY"):
        value = os.getenv(variable, "")
        if value:
            message = message.replace(value, "[redacted]")
    return {"http_status": response.status_code, "message": message[:500]}


async def configure(create=False, voice_id="21m00Tcm4TlvDq8ikWAM"):
    load_config()
    preview = PROJECT_ROOT / ".local/elevenlabs_agent_config.json"
    preview.parent.mkdir(exist_ok=True)
    preview.write_text(json.dumps(agent_configuration(voice_id), indent=2), encoding="utf-8")
    if not create:
        return {"status": "preview", "configuration": str(preview)}
    if os.getenv("ELEVENLABS_AGENT_ID"):
        return {"status": "already_configured", "agent_id": os.getenv("ELEVENLABS_AGENT_ID")}
    async with httpx.AsyncClient(timeout=30, headers={"xi-api-key": os.getenv("ELEVENLABS_API_KEY", "")}) as client:
        existing = await client.get("https://api.elevenlabs.io/v1/convai/agents")
        if not existing.is_success:
            return {"status": "provider_error", **safe_error(existing)}
        matches = [a for a in existing.json().get("agents", []) if a.get("name") == AGENT_NAME]
        if len(matches) > 1:
            return {"status": "ambiguous", "message": "Multiple matching agents; set ELEVENLABS_AGENT_ID explicitly."}
        if matches:
            agent_id = matches[0]["agent_id"]
            status = "reused"
        else:
            # The standard Rachel voice avoids requiring voice-library read access.
            config = agent_configuration(voice_id)
            preview.write_text(json.dumps(config, indent=2), encoding="utf-8")
            response = await client.post("https://api.elevenlabs.io/v1/convai/agents/create", json=config)
            if not response.is_success:
                return {"status": "provider_error", **safe_error(response)}
            agent_id, status = response.json()["agent_id"], "created"
        set_key(PROJECT_ROOT / ".env", "ELEVENLABS_AGENT_ID", agent_id)
        return {"status": status, "agent_id": agent_id}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create", action="store_true", help="Create the agent if no dedicated agent is configured.")
    parser.add_argument("--voice-id", default="21m00Tcm4TlvDq8ikWAM")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(configure(args.create, args.voice_id)), indent=2))
