"""Backend-first conversation service. Run: uvicorn conversation_app:app."""
from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import logging
import os
import secrets
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from src.conversation_acoustics import RollingAudio, extract_acoustics, SAMPLE_RATE
from src.conversation_classifier import AcousticClassifier
from src.elevenlabs_agent import ElevenLabsAgent
from src.config import load_config
from src.gemini_reporter import GeminiReporter

logger = logging.getLogger(__name__)
TASKS = {"conversation", "reading", "sustained_vowel"}


def create_app(classifier=None, agent=None, reporter=None):
    load_config()
    application = FastAPI(title="Voice PD conversational classifier")
    classifier = classifier or AcousticClassifier(os.environ.get(
        "PD_MODEL_DIR", str(Path(__file__).parent / "models_conversation")))
    agent = agent or ElevenLabsAgent(os.environ.get("ELEVENLABS_API_KEY", ""),
                                    os.environ.get("ELEVENLABS_AGENT_ID", ""))
    reporter = reporter or GeminiReporter(os.environ.get("GEMINI_API_KEY", ""),
                                         os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite"))
    analysis_slots = asyncio.Semaphore(2)

    @application.get("/health")
    def health():
        return {"ok": True, "elevenlabs_configured": agent.configured,
                "gemini_configured": reporter.configured,
                "classifier_loaded": classifier.pipeline is not None,
                "classifier_error": classifier.error,
                "tasks": classifier.metadata.get("tasks", []), "minimum_age": 50}

    @application.websocket("/api/conversation")
    async def conversation(client: WebSocket):
        origins = os.environ.get("PD_ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
        token = os.environ.get("PD_API_TOKEN", "")
        # Native clients have no Origin header. Browser clients must match.
        if client.headers.get("origin") and client.headers["origin"] not in origins:
            await client.close(code=1008)
            return
        if token and not secrets.compare_digest(client.headers.get("authorization", ""), f"Bearer {token}"):
            await client.close(code=1008)
            return
        await client.accept()
        try:
            start = await asyncio.wait_for(client.receive_json(), 15)
            if (not isinstance(start, dict) or start.get("type") != "start" or
                type(start.get("age")) is not int or not 0 <= start["age"] <= 120 or
                start.get("task", "conversation") not in TASKS):
                raise ValueError("First message must be start with integer age (0-120) and a supported task.")
            if not agent.configured:
                await client.send_json({"type": "error", "code": "elevenlabs_not_configured",
                    "message": "Set ELEVENLABS_API_KEY and ELEVENLABS_AGENT_ID on the server."})
                return
            async with agent.conversation(start["age"], start.get("task", "conversation")) as upstream:
                # Bound each conversation to ten minutes; idle microphone to 45 seconds.
                async with asyncio.timeout(600):
                    await run_session(client, upstream, classifier, start, analysis_slots, reporter)
        except WebSocketDisconnect:
            pass
        except (ValueError, binascii.Error) as exc:
            await client.send_json({"type": "error", "code": "invalid_message", "message": str(exc)})
        except TimeoutError:
            with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                await client.send_json({"type": "error", "code": "session_timeout", "message": "Conversation timed out."})
        except Exception:
            # Avoid logging signed URLs, API keys or raw speech/transcripts.
            logger.error("Conversation failed; check provider configuration and connectivity.")
            with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                await client.send_json({"type": "error", "code": "conversation_failed",
                    "message": "Could not complete conversation. Check provider configuration and connectivity."})
        finally:
            with contextlib.suppress(RuntimeError):
                await client.close()

    return application


async def run_session(client, upstream, classifier, start, analysis_slots, reporter=None):
    audio = RollingAudio()
    pending = asyncio.Queue(maxsize=1)
    send_lock = asyncio.Lock()
    latest = None
    sequence = 0
    epoch = 0
    current_task = start.get("task", "conversation")
    ready = asyncio.Event()
    report_cache = None
    tasks = []

    async def emit(event):
        async with send_lock:
            await client.send_json(event)

    async def generate_report():
        nonlocal report_cache
        snapshot = latest
        snapshot_epoch = epoch
        if reporter is None:
            return {"status": "disabled", "text": "Reports are not configured."}
        if report_cache is not None and report_cache[0] == (snapshot or {}).get("sequence"):
            return report_cache[1]
        result = await reporter.generate(snapshot)
        if snapshot_epoch != epoch:
            return {"status": "unavailable", "text": "The analysis changed during report generation. Request a report for the current result."}
        report_cache = ((snapshot or {}).get("sequence"), result)
        return result

    def reset_window(task=None):
        nonlocal epoch, latest, current_task
        if task is not None:
            if task not in TASKS:
                raise ValueError("Unsupported task.")
            current_task = task
            epoch += 1
            latest = None
        audio.clear()
        # A capture gap does not invalidate an already clean speech window.
        # A task switch invalidates queued and in-flight work from the old task.
        if task is not None and not pending.empty():
            pending.get_nowait()
            pending.task_done()

    async def provider_events():
        async for raw in upstream:
            event = json.loads(raw)
            if event.get("type") == "ping":
                await upstream.send(json.dumps({"type": "pong", "event_id": event["ping_event"]["event_id"]}))
                continue
            if event.get("type") == "client_tool_call":
                tool = event["client_tool_call"]
                failed = False
                try:
                    if tool["tool_name"] == "set_recording_task":
                        requested_task = tool.get("parameters", {}).get("task")
                        if requested_task not in TASKS:
                            raise ValueError("A supported task parameter is required.")
                        reset_window(requested_task)
                        result = {"task": current_task, "status": "collecting"}
                        await emit({"type": "task_changed", **result})
                    elif tool["tool_name"] == "get_voice_analysis":
                        result = latest or {"status": "collecting", "task": current_task}
                    else:
                        raise ValueError("Unsupported client tool.")
                except (ValueError, TypeError, AttributeError) as exc:
                    failed, result = True, {"error": str(exc)}
                await upstream.send(json.dumps({"type": "client_tool_result",
                    "tool_call_id": tool["tool_call_id"], "is_error": failed,
                    "result": json.dumps(result)}))
                continue
            if event.get("type") == "conversation_initiation_metadata":
                metadata = event["conversation_initiation_metadata_event"]
                if metadata.get("user_input_audio_format") != "pcm_16000":
                    raise ValueError("Configure ElevenLabs input audio as pcm_16000.")
                await emit(event)
                await upstream.send(json.dumps({"type": "contextual_update", "text": json.dumps({
                    "supported_recording_tasks": classifier.metadata.get("tasks", []),
                    "classifier_available": classifier.pipeline is not None})}))
                ready.set()
                continue
            # Playback, transcript and interruption events are forwarded intact.
            # Only client microphone audio enters the measurement path.
            await emit(event)
        raise RuntimeError("ElevenLabs disconnected.")

    async def measure_windows():
        nonlocal latest, sequence
        while True:
            y, end_seconds, task, window_epoch = await pending.get()
            try:
                started = time.perf_counter()
                async with analysis_slots:
                    measurements = await asyncio.to_thread(extract_acoustics, y)
                    risk = await asyncio.to_thread(classifier.predict, measurements, start["age"], task)
                if window_epoch != epoch:
                    continue
                sequence += 1
                latest = {"type": "analysis", "sequence": sequence, "task": task,
                          "window_start_seconds": end_seconds - 4,
                          "window_end_seconds": end_seconds, **measurements,
                          "classification": risk, "processing_ms": round(1000 * (time.perf_counter() - started), 1)}
                await emit(latest)
                await upstream.send(json.dumps({"type": "contextual_update", "text": json.dumps({
                    "voice_quality": measurements["quality"], "classifier": risk})}))
            finally:
                pending.task_done()

    async def client_audio():
        nonlocal epoch, current_task, latest
        await asyncio.wait_for(ready.wait(), 20)
        await emit({"type": "ready", "sample_rate": SAMPLE_RATE, "encoding": "pcm_s16le",
                    "window_seconds": 4, "hop_seconds": 1,
                    "age_coverage": "eligible" if start["age"] >= 50 else "outside_evaluated_population"})
        while True:
            packet = await asyncio.wait_for(client.receive(), 45)
            if packet["type"] == "websocket.disconnect":
                raise WebSocketDisconnect()
            if packet.get("bytes") is not None:
                pcm = packet["bytes"]
            else:
                if len(packet.get("text", "")) > 48000:
                    raise ValueError("Message too large.")
                event = json.loads(packet.get("text", ""))
                if not isinstance(event, dict):
                    raise ValueError("Expected a JSON object.")
                if event.get("type") == "end":
                    await asyncio.wait_for(pending.join(), 15)
                    report = await generate_report()
                    await emit({"type": "summary", "latest_analysis": latest,
                                "completed_windows": sequence, "report": report})
                    return
                if event.get("type") == "report":
                    await emit({"type": "report", **await generate_report()})
                    continue
                if event.get("type") in {"set_task", "reset_audio"}:
                    if event.get("type") == "set_task":
                        if event.get("task") not in TASKS:
                            raise ValueError("Unsupported task.")
                        reset_window(event["task"])
                        await upstream.send(json.dumps({"type": "contextual_update", "text": f"Recording task: {current_task}."}))
                    else:
                        reset_window()
                    continue
                if event.get("type") != "audio" or not isinstance(event.get("audio"), str):
                    raise ValueError("Expected audio, set_task, reset_audio, report or end.")
                pcm = base64.b64decode(event["audio"], validate=True)
            window = audio.append(pcm)
            await upstream.send(json.dumps({"user_audio_chunk": base64.b64encode(pcm).decode("ascii")}))
            if window is not None:
                if pending.full():
                    pending.get_nowait()
                    pending.task_done()
                pending.put_nowait((*window, current_task, epoch))

    tasks = [asyncio.create_task(provider_events()), asyncio.create_task(measure_windows()),
             asyncio.create_task(client_audio())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for finished in done:
            finished.result()
    finally:
        for running in tasks:
            running.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


app = create_app()
