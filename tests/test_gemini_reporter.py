import asyncio
import json
import httpx
import pytest
from src.gemini_reporter import GeminiReporter
from dotenv import load_dotenv


def analysis():
    return {"sequence": 7, "task": "sustained_vowel", "features": {"cpps_db": 12},
            "units": {"cpps_db": "dB"}, "quality": {"usable": True},
            "classification": {"status": "experimental", "score_pd": .42},
            "transcript": "private", "audio": "private"}


def test_report_sends_only_allowlisted_evidence_and_uses_header_key():
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [
            {"text": "internal chain", "thought": True}, {"text": "The score is experimental."}]}}]})
    reporter = GeminiReporter("secret-test-key", transport=httpx.MockTransport(handler))
    result = asyncio.run(reporter.generate(analysis()))
    assert result["status"] == "generated"
    assert result["analysis_sequence"] == 7
    assert result["text"] == "The score is experimental."
    assert requests[0].headers["x-goog-api-key"] == "secret-test-key"
    assert "secret-test-key" not in str(requests[0].url)
    body = json.loads(requests[0].content)
    evidence = json.loads(body["contents"][0]["parts"][0]["text"])
    assert evidence["classification"]["score_pd"] == .42
    assert "transcript" not in evidence and "audio" not in evidence
    assert "not a personal probability" in body["systemInstruction"]["parts"][0]["text"]


@pytest.mark.parametrize("code", [400, 401, 403, 429, 503])
def test_provider_failure_keeps_evidence_and_redacts_raw_error(code):
    transport = httpx.MockTransport(lambda req: httpx.Response(code, json={"error": {"message": "secret-test-key"}}))
    result = asyncio.run(GeminiReporter("secret-test-key", transport=transport).generate(analysis()))
    assert result["status"] == "provider_error"
    assert result["http_status"] == code
    assert result["evidence"]["classification"]["score_pd"] == .42
    assert "secret-test-key" not in json.dumps(result)


def test_empty_generation_missing_key_and_missing_analysis():
    reporter = GeminiReporter("key", transport=httpx.MockTransport(lambda req: httpx.Response(200, json={})))
    assert asyncio.run(reporter.generate(analysis()))["status"] == "empty_response"
    assert asyncio.run(reporter.generate(None))["status"] == "unavailable"
    assert asyncio.run(GeminiReporter("").generate(analysis()))["status"] == "disabled"
    with pytest.raises(ValueError):
        GeminiReporter("key", "../../invalid")


def test_environment_overrides_local_env(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("EXAMPLE_LOCAL_SETTING=local\n")
    monkeypatch.setenv("EXAMPLE_LOCAL_SETTING", "caller")
    load_dotenv(path, override=False)
    import os
    assert os.environ["EXAMPLE_LOCAL_SETTING"] == "caller"
