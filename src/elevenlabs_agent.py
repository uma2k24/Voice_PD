"""Server-side ElevenLabs Agents WebSocket adapter (no keys sent to clients)."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
import httpx
from websockets.asyncio.client import connect

AGENT_PROMPT = """You are a calm voice research assistant helping collect speech
for a Parkinson's voice screening prototype. This is not a diagnostic service.
Have a natural conversation, asking one short question at a time about the
user's day or hobbies. Encourage answers in complete sentences. Do not infer
disease from what the user says. Follow the recording task supplied in context:
conversation means spontaneous conversation; reading means ask them to read
'The sun rose over the quiet town. I walked to the market and bought fresh fruit.';
sustained_vowel means ask them to hold a comfortable 'ah' for at least five
seconds, then rest. Never ask them to strain their voice. If recording quality
is poor, ask them to move to a quiet room and try again. The server may supply
structured classifier status and measurements. Only explain the supplied status.
A score is experimental, not the probability that this person has Parkinson's.
Do not invent thresholds, results, diagnoses, or medical advice. Users below 50
are outside the evaluated population; this does not mean they are healthy.
After the user agrees, use the set_recording_task client tool before collecting
each speech task. If the server says conversation is unsupported but
sustained_vowel is supported, invite them to do the comfortable 'ah' exercise
during your conversation and call set_recording_task with sustained_vowel before
giving the instruction. Wait for the tool result. Keep sustained_vowel active
until a completed analysis is retrieved, then switch back to conversation.
Never tag ordinary speech as a sustained vowel. The
get_voice_analysis tool returns the latest completed measurement and status;
use it after an exercise instead of guessing a result. If still collecting,
wait and ask again only after the user has had time to produce the sample.
Stop promptly if the user asks to stop."""
FIRST_MESSAGE = "Hi, I'm your voice research assistant. We'll collect a short speech sample together. This screening cannot diagnose Parkinson's. Are you comfortable starting?"


class ElevenLabsAgent:
    def __init__(self, api_key: str, agent_id: str):
        self.api_key, self.agent_id = api_key, agent_id

    @property
    def configured(self):
        return bool(self.api_key and self.agent_id)

    @asynccontextmanager
    async def conversation(self, age: int, task: str):
        if not self.configured:
            raise RuntimeError("Set ELEVENLABS_API_KEY and ELEVENLABS_AGENT_ID.")
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                "https://api.elevenlabs.io/v1/convai/conversation/get-signed-url",
                params={"agent_id": self.agent_id}, headers={"xi-api-key": self.api_key})
            response.raise_for_status()
            url = response.json()["signed_url"]
        # Authenticated URL stays on this server, including its bearer token.
        async with connect(url, open_timeout=15, max_size=2**20, max_queue=8) as socket:
            await socket.send(json.dumps({
                "type": "conversation_initiation_client_data",
                "conversation_config_override": {"agent": {
                    "prompt": {"prompt": AGENT_PROMPT}, "first_message": FIRST_MESSAGE}},
                "dynamic_variables": {"user_age": age, "recording_task": task}}))
            await socket.send(json.dumps({"type": "contextual_update",
                "text": f"Recording task: {task}. Age coverage: {'eligible' if age >= 50 else 'outside evaluated population'}."}))
            yield socket
