from contextlib import asynccontextmanager
import asyncio
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import conversation_app
from src.conversation_acoustics import (FEATURE_NAMES, EXTRACTOR_VERSION, SETTINGS,
    RollingAudio, extract_acoustics, SAMPLE_RATE)
from src.conversation_classifier import AcousticClassifier
from scripts.train_conversation_classifier import fit_baseline, read_manifest


@pytest.fixture(autouse=True)
def isolate_local_credentials(monkeypatch):
    # Unit tests must never use credentials from the developer's local .env.
    for key in ("ELEVENLABS_API_KEY", "ELEVENLABS_AGENT_ID", "GEMINI_API_KEY", "PD_API_TOKEN"):
        monkeypatch.setenv(key, "")


def vowel(duration=4):
    t = np.arange(int(SAMPLE_RATE * duration)) / SAMPLE_RATE
    return sum(.3 / h * np.sin(2 * np.pi * 150 * h * t) for h in range(1, 8))


def test_acoustics_match_praat_and_units():
    from parselmouth import Sound
    from parselmouth.praat import call
    y = vowel()
    result = extract_acoustics(y)
    assert result["quality"]["usable"]
    f = result["features"]
    assert f["f0_mean_hz"] == pytest.approx(150, abs=.1)
    assert f["mean_period_seconds"] == pytest.approx(1 / 150, abs=1e-5)
    assert 0 <= f["gne_ratio"] <= 1
    sound = Sound(y - y.mean(), SAMPLE_RATE)
    cepstrogram = call(sound, "To PowerCepstrogram", 60, .002, 5000, 50)
    expected = call(cepstrogram, "Get CPPS", False, .01, .001, 60, 500, .05,
                    "Parabolic", .001, 0, "Straight", "Robust")
    assert f["cpps_db"] == pytest.approx(expected)
    pitch = sound.to_pitch_ac(time_step=.01, pitch_floor=60, pitch_ceiling=500)
    pulses = call([sound, pitch], "To PointProcess (cc)")
    assert f["jitter_local_percent"] == pytest.approx(100 * call(
        pulses, "Get jitter (local)", 0, 0, 1 / 500, 1 / 60, 1.3))
    assert f["shimmer_local_percent"] == pytest.approx(100 * call(
        [sound, pulses], "Get shimmer (local)", 0, 0, 1 / 500, 1 / 60, 1.3, 1.6))
    json.dumps(result, allow_nan=False)


def test_silence_noise_and_clipping_never_get_usable_scores():
    assert not extract_acoustics(np.zeros(64000))["quality"]["usable"]
    assert not extract_acoustics(np.random.default_rng(42).normal(0, .1, 64000))["quality"]["usable"]
    assert "clipping" in extract_acoustics(np.clip(vowel() * 6, -1, 1))["quality"]["reasons"]
    with pytest.raises(ValueError):
        extract_acoustics(np.array([np.nan]))


def test_bounded_rolling_windows_and_reset():
    audio = RollingAudio()
    pcm = (vowel(1) * 32767).astype("<i2").tobytes()
    for _ in range(3):
        assert audio.append(pcm) is None
    assert audio.append(pcm)[1] == 4
    assert audio.append(pcm)[1] == 5
    assert audio.samples.size == 64000
    audio.clear()
    assert audio.append(pcm) is None
    for bad in (b"", b"\x00", bytes(32002)):
        with pytest.raises(ValueError):
            audio.append(bad)


def feature_frame():
    rng = np.random.default_rng(12)
    rows = []
    for speaker in range(12):
        for recording in range(2):
            rows.append({"speaker_id": str(speaker), "label": speaker % 2,
                         **dict(zip(FEATURE_NAMES, rng.normal(speaker % 2, .5, len(FEATURE_NAMES))))})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("classifier", ["logreg", "random_forest"])
def test_speaker_cv_and_model_roundtrip(tmp_path, classifier):
    frame = feature_frame()
    frame.loc[0, "f1_mean_hz"] = np.nan
    pipeline, metrics, speakers, splits = fit_baseline(frame, classifier=classifier, folds=3)
    assert metrics["n_speakers"] == len(speakers) == 12
    for fold in splits:
        assert not set(fold["train"]) & set(fold["test"])
    joblib.dump(pipeline, tmp_path / "classifier.joblib")
    metadata = {"extractor_version": EXTRACTOR_VERSION, "features": FEATURE_NAMES,
                "settings": SETTINGS, "minimum_age": 50, "tasks": ["conversation"], "classifier": classifier}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    model = AcousticClassifier(tmp_path)
    measured = {"quality": {"usable": True}, "features": frame.iloc[0][FEATURE_NAMES].to_dict()}
    assert 0 <= model.predict(measured, 60, "conversation")["score_pd"] <= 1
    assert model.predict(measured, 49, "conversation")["score_pd"] is None
    assert model.predict(measured, 60, "reading")["status"] == "unsupported_recording_task"
    measured["quality"]["usable"] = False
    assert model.predict(measured, 60, "conversation")["score_pd"] is None
    metadata["extractor_version"] = "other"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    assert AcousticClassifier(tmp_path).pipeline is None


def test_manifest_rejects_inconsistent_speakers_and_excludes_under_50(tmp_path):
    frame = pd.DataFrame([{"path": f"{n}.wav", "speaker_id": str(n), "label": n % 2,
                          "age": 40 if n == 0 else 60, "sex": "M", "task": "reading"} for n in range(4)])
    path = tmp_path / "manifest.csv"
    frame.to_csv(path, index=False)
    assert len(read_manifest(path)) == 3
    frame.loc[2, "speaker_id"] = "1"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Inconsistent label"):
        read_manifest(path)


class FakeSocket:
    def __init__(self):
        self.events = asyncio.Queue()
        self.sent = []
        self.events.put_nowait({"type": "conversation_initiation_metadata",
            "conversation_initiation_metadata_event": {"user_input_audio_format": "pcm_16000",
                "agent_output_audio_format": "pcm_16000", "conversation_id": "test"}})
        self.events.put_nowait({"type": "ping", "ping_event": {"event_id": 1}})
        self.events.put_nowait({"type": "audio", "audio_event": {"audio_base_64": "AAA="}})

    def __aiter__(self):
        return self

    async def __anext__(self):
        return json.dumps(await self.events.get())

    async def send(self, raw):
        self.sent.append(json.loads(raw))


class FakeAgent:
    configured = True

    @asynccontextmanager
    async def conversation(self, age, task):
        self.socket = FakeSocket()
        yield self.socket


def test_websocket_audio_relay_analysis_and_shutdown(tmp_path, monkeypatch):
    fake = FakeAgent()
    model = AcousticClassifier(tmp_path)
    # Exercise scheduling/relay with deterministic extraction; actual Praat is tested above.
    monkeypatch.setattr(conversation_app, "extract_acoustics", lambda y: {
        "features": dict.fromkeys(FEATURE_NAMES, 1), "quality": {"usable": True}})
    with TestClient(conversation_app.create_app(model, fake)) as client:
        with client.websocket_connect("/api/conversation") as ws:
            ws.send_json({"type": "start", "age": 49, "task": "conversation"})
            events = []
            while not events or events[-1]["type"] != "ready":
                events.append(ws.receive_json())
            assert any(e["type"] == "audio" for e in events)
            pcm = (vowel(1) * 32767).astype("<i2").tobytes()
            for _ in range(4):
                ws.send_bytes(pcm)
            analysis = ws.receive_json()
            assert analysis["type"] == "analysis"
            assert analysis["classification"]["status"] == "outside_evaluated_population"
            ws.send_json({"type": "end"})
            assert ws.receive_json()["type"] == "summary"
    assert {"type": "pong", "event_id": 1} in fake.socket.sent
    assert len([e for e in fake.socket.sent if "user_audio_chunk" in e]) == 4


def test_missing_provider_bad_start_and_auth(tmp_path, monkeypatch):
    from src.elevenlabs_agent import ElevenLabsAgent
    app = conversation_app.create_app(AcousticClassifier(tmp_path), ElevenLabsAgent("", ""))
    with TestClient(app) as client:
        assert client.get("/health").json()["classifier_loaded"] is False
        with client.websocket_connect("/api/conversation") as ws:
            ws.send_json({"type": "start", "age": 60})
            assert ws.receive_json()["code"] == "elevenlabs_not_configured"
        with client.websocket_connect("/api/conversation") as ws:
            ws.send_json({"type": "start", "age": True})
            assert ws.receive_json()["code"] == "invalid_message"
        monkeypatch.setenv("PD_API_TOKEN", "secret-test")
        with pytest.raises(Exception):
            with client.websocket_connect("/api/conversation"):
                pass


def test_agent_tools_select_actual_task_and_report_collecting(tmp_path, monkeypatch):
    class ToolAgent(FakeAgent):
        @asynccontextmanager
        async def conversation(self, age, task):
            self.socket = FakeSocket()
            for name, params, ident in [("set_recording_task", {"task": "sustained_vowel"}, "set"),
                                        ("get_voice_analysis", {}, "get"),
                                        ("set_recording_task", {}, "bad")]:
                self.socket.events.put_nowait({"type": "client_tool_call", "client_tool_call": {
                    "tool_name": name, "parameters": params, "tool_call_id": ident}})
            yield self.socket
    fake = ToolAgent()
    monkeypatch.setattr(conversation_app, "extract_acoustics", lambda y: {
        "features": dict.fromkeys(FEATURE_NAMES, 1), "quality": {"usable": True}})
    with TestClient(conversation_app.create_app(AcousticClassifier(tmp_path), fake)) as client:
        with client.websocket_connect("/api/conversation") as ws:
            ws.send_json({"type": "start", "age": 60})
            while ws.receive_json()["type"] != "ready":
                pass
            for _ in range(4):
                ws.send_bytes(bytes(32000))
            while True:
                event = ws.receive_json()
                if event["type"] == "analysis":
                    assert event["task"] == "sustained_vowel"
                    break
            ws.send_json({"type": "end"})
            assert ws.receive_json()["type"] == "summary"
    results = {e["tool_call_id"]: e for e in fake.socket.sent if e.get("type") == "client_tool_result"}
    assert json.loads(results["get"]["result"])["status"] == "collecting"
    assert results["bad"]["is_error"]
    assert not results["set"]["is_error"]


def test_task_switch_discards_inflight_analysis_and_summary(tmp_path, monkeypatch):
    import threading
    started, release = threading.Event(), threading.Event()
    def slow_extract(y):
        started.set()
        assert release.wait(timeout=3)
        return {"features": dict.fromkeys(FEATURE_NAMES, 1), "quality": {"usable": True}}
    monkeypatch.setattr(conversation_app, "extract_acoustics", slow_extract)
    with TestClient(conversation_app.create_app(AcousticClassifier(tmp_path), FakeAgent())) as client:
        with client.websocket_connect("/api/conversation") as ws:
            ws.send_json({"type": "start", "age": 60})
            while ws.receive_json()["type"] != "ready":
                pass
            for _ in range(4):
                ws.send_bytes(bytes(32000))
            assert started.wait(timeout=3)
            ws.send_json({"type": "set_task", "task": "reading"})
            ws.send_json({"type": "end"})
            release.set()
            summary = ws.receive_json()
            assert summary["type"] == "summary"
            assert summary["latest_analysis"] is None


def test_playback_capture_gap_preserves_clean_analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(conversation_app, "extract_acoustics", lambda y: {
        "features": dict.fromkeys(FEATURE_NAMES, 1), "quality": {"usable": True}})
    with TestClient(conversation_app.create_app(AcousticClassifier(tmp_path), FakeAgent())) as client:
        with client.websocket_connect("/api/conversation") as ws:
            ws.send_json({"type": "start", "age": 60})
            while ws.receive_json()["type"] != "ready":
                pass
            for _ in range(4):
                ws.send_bytes(bytes(32000))
            analysis = ws.receive_json()
            assert analysis["type"] == "analysis"
            ws.send_json({"type": "reset_audio"})
            ws.send_json({"type": "end"})
            assert ws.receive_json()["latest_analysis"] == analysis


@pytest.mark.parametrize("report_status", ["generated", "provider_error"])
def test_report_uses_server_measurement_and_summary_reuses_it(tmp_path, monkeypatch, report_status):
    class Reporter:
        configured = True
        def __init__(self):
            self.calls = []
        async def generate(self, measurement):
            self.calls.append(measurement)
            return {"status": report_status, "analysis_sequence": measurement["sequence"],
                    "text": "Research explanation", "evidence": measurement}
    reporter = Reporter()
    monkeypatch.setattr(conversation_app, "extract_acoustics", lambda y: {
        "features": dict.fromkeys(FEATURE_NAMES, 1), "quality": {"usable": True}})
    with TestClient(conversation_app.create_app(AcousticClassifier(tmp_path), FakeAgent(), reporter)) as client:
        assert client.get("/health").json()["gemini_configured"]
        with client.websocket_connect("/api/conversation") as ws:
            ws.send_json({"type": "start", "age": 60})
            while ws.receive_json()["type"] != "ready":
                pass
            for _ in range(4):
                ws.send_bytes(bytes(32000))
            measured = ws.receive_json()
            ws.send_json({"type": "report", "classification": {"score_pd": 1}})
            report = ws.receive_json()
            assert report["type"] == "report"
            assert report["evidence"]["classification"]["score_pd"] is None
            ws.send_json({"type": "end"})
            summary = ws.receive_json()
            assert summary["latest_analysis"] == measured
            assert summary["report"]["analysis_sequence"] == measured["sequence"]
            assert summary["report"]["status"] == report_status
    assert len(reporter.calls) == 1


def test_elevenlabs_adapter_uses_signed_url_and_protocol(monkeypatch):
    from src import elevenlabs_agent as module
    calls = []
    socket = FakeSocket()
    class HttpClient:
        def __init__(self, **kwargs):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def get(self, url, **kwargs):
            calls.append((url, kwargs))
            class Response:
                def raise_for_status(self):
                    pass
                def json(self):
                    return {"signed_url": "wss://example.test/private-token"}
            return Response()
    @asynccontextmanager
    async def connection(url, **kwargs):
        assert url == "wss://example.test/private-token"
        yield socket
    monkeypatch.setattr(module.httpx, "AsyncClient", HttpClient)
    monkeypatch.setattr(module, "connect", connection)
    async def check():
        async with module.ElevenLabsAgent("test-key", "test-agent").conversation(60, "reading") as result:
            assert result is socket
    asyncio.run(check())
    assert calls[0][1]["headers"] == {"xi-api-key": "test-key"}
    assert socket.sent[0]["type"] == "conversation_initiation_client_data"
    assert socket.sent[0]["dynamic_variables"]["recording_task"] == "reading"
    assert "test-key" not in json.dumps(socket.sent)


@pytest.mark.parametrize("bad", [{"type": "audio", "audio": "?"}, {"type": "audio", "audio": "AA=="}, []])
def test_invalid_audio_packets_are_rejected(tmp_path, bad):
    with TestClient(conversation_app.create_app(AcousticClassifier(tmp_path), FakeAgent())) as client:
        with client.websocket_connect("/api/conversation") as ws:
            ws.send_json({"type": "start", "age": 60})
            while ws.receive_json()["type"] != "ready":
                pass
            ws.send_json(bad)
            assert ws.receive_json()["code"] == "invalid_message"
