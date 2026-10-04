"""Verify a real ElevenLabs handshake and greeting without recording a person."""
import asyncio
import json
import os
from src.config import load_config
from src.elevenlabs_agent import ElevenLabsAgent


async def verify():
    load_config()
    agent = ElevenLabsAgent(os.getenv("ELEVENLABS_API_KEY", ""), os.getenv("ELEVENLABS_AGENT_ID", ""))
    result = {"status": "unavailable", "metadata_received": False, "audio_received": False}
    if not agent.configured:
        return result
    try:
        async with asyncio.timeout(25):
            async with agent.conversation(60, "conversation") as socket:
                async for raw in socket:
                    event = json.loads(raw)
                    if event.get("type") == "ping":
                        await socket.send(json.dumps({"type": "pong", "event_id": event["ping_event"]["event_id"]}))
                    elif event.get("type") == "conversation_initiation_metadata":
                        metadata = event["conversation_initiation_metadata_event"]
                        result.update(metadata_received=True,
                                      input_format=metadata.get("user_input_audio_format"),
                                      output_format=metadata.get("agent_output_audio_format"))
                    elif event.get("type") == "audio":
                        result["audio_received"] = bool(event["audio_event"].get("audio_base_64"))
                        if result["metadata_received"] and result["audio_received"]:
                            result["status"] = "connected"
                            return result
                    elif event.get("type") in {"client_error", "error"}:
                        result["status"] = "provider_error"
                        return result
    except Exception as exc:
        # Do not print exception messages that may contain the signed URL.
        result.update(status="connection_failed", error_type=type(exc).__name__)
    return result


if __name__ == "__main__":
    print(json.dumps(asyncio.run(verify()), indent=2))
