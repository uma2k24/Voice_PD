"""Microphone/speaker client without a frontend. Use headphones to avoid echo.

python -m scripts.conversation_cli --age 60 --task conversation
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import os
import threading
import time

from websockets.asyncio.client import connect
from src.config import load_config


async def run(url, age, task):
    load_config()
    import sounddevice as sd
    playback = bytearray()
    playback_lock = threading.Lock()
    last_played = 0.0
    stop = asyncio.Event()
    headers = {"Authorization": f"Bearer {os.environ['PD_API_TOKEN']}"} if os.environ.get("PD_API_TOKEN") else None

    def speaker_callback(outdata, frames, callback_time, status):
        nonlocal last_played
        with playback_lock:
            count = min(len(outdata), len(playback))
            outdata[:] = bytes(playback[:count]) + bytes(len(outdata) - count)
            del playback[:count]
            if count:
                last_played = time.monotonic()

    async with connect(url, additional_headers=headers, max_size=2**20) as socket:
        await socket.send(json.dumps({"type": "start", "age": age, "task": task}))
        ready = asyncio.Event()

        async def receive():
            async for raw in socket:
                event = json.loads(raw)
                kind = event.get("type")
                if kind == "conversation_initiation_metadata":
                    if event["conversation_initiation_metadata_event"].get("agent_output_audio_format") != "pcm_16000":
                        raise ValueError("Set ElevenLabs agent output format to pcm_16000.")
                elif kind == "ready":
                    ready.set()
                    print("Connected. Speak naturally. Ctrl+C to finish.")
                elif kind == "audio":
                    audio = base64.b64decode(event["audio_event"]["audio_base_64"])
                    with playback_lock:
                        if len(playback) + len(audio) > 16000 * 2 * 30:
                            raise ValueError("Agent playback exceeded 30 seconds of buffered audio.")
                        playback.extend(audio)
                elif kind == "interruption":
                    with playback_lock:
                        playback.clear()
                elif kind == "analysis":
                    print(json.dumps({"features": event["features"], "quality": event["quality"],
                                      "classification": event["classification"]}, indent=2))
                elif kind in {"user_transcript", "agent_response"}:
                    print(json.dumps(event))
                elif kind == "report":
                    print(json.dumps(event, indent=2))
                elif kind in {"summary", "error"}:
                    print(json.dumps(event, indent=2))
                    stop.set()
                    return

        async def microphone():
            await ready.wait()
            was_playing = False
            with sd.RawInputStream(samplerate=16000, channels=1, dtype="int16", blocksize=1600) as stream:
                while not stop.is_set():
                    data, overflow = await asyncio.to_thread(stream.read, 1600)
                    with playback_lock:
                        playing = bool(playback) or time.monotonic() - last_played < .3
                    if playing != was_playing or overflow:
                        await socket.send(json.dumps({"type": "reset_audio"}))
                        was_playing = playing
                    # Half-duplex reference client: never analyse speaker playback.
                    if not playing and not overflow:
                        await socket.send(bytes(data))

        with sd.RawOutputStream(samplerate=16000, channels=1, dtype="int16", blocksize=1600,
                                callback=speaker_callback):
            receiver = asyncio.create_task(receive())
            sender = asyncio.create_task(microphone())
            try:
                done, _ = await asyncio.wait([receiver, sender], return_when=asyncio.FIRST_COMPLETED)
                for finished in done:
                    finished.result()
            finally:
                sender.cancel()
                with contextlib.suppress(Exception):
                    await socket.send(json.dumps({"type": "end"}))
                if not receiver.done():
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(receiver, 40)
                receiver.cancel()
                await asyncio.gather(sender, receiver, return_exceptions=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8000/api/conversation")
    parser.add_argument("--age", type=int, required=True)
    parser.add_argument("--task", choices=["conversation", "reading", "sustained_vowel"], default="conversation")
    args = parser.parse_args()
    if not 0 <= args.age <= 120:
        parser.error("age must be in 0-120")
    try:
        asyncio.run(run(args.url, args.age, args.task))
    except KeyboardInterrupt:
        print("Conversation ended.")


if __name__ == "__main__":
    main()
