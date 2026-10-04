"""Read-only connectivity checks; never print credentials or signed URLs."""
import asyncio
import json
import os
import httpx
import argparse
from src.config import load_config
from src.gemini_reporter import GeminiReporter


async def check_access():
    load_config()
    async with httpx.AsyncClient(timeout=20) as client:
        async def check(name, url, header):
            try:
                response = await client.get(url, headers=header)
                payload = response.json()
                result = {"provider": name, "http_status": response.status_code}
                if response.is_success:
                    if name == "ElevenLabs":
                        result["agents"] = [{"agent_id": a["agent_id"], "name": a.get("name")}
                                            for a in payload.get("agents", [])]
                    else:
                        result["generation_models"] = [m["name"] for m in payload.get("models", [])
                            if "generateContent" in m.get("supportedGenerationMethods", [])][:20]
                else:
                    if name == "Gemini":
                        result["error_status"] = payload.get("error", {}).get("status")
                        result["error_reasons"] = [d.get("reason") for d in payload.get("error", {}).get("details", []) if d.get("reason")]
                    else:
                        detail = payload.get("detail", {})
                        result["error_status"] = detail.get("status") if isinstance(detail, dict) else "request_rejected"
                        message = detail.get("message", "") if isinstance(detail, dict) else ""
                        for key in (os.getenv("ELEVENLABS_API_KEY", ""), os.getenv("GEMINI_API_KEY", "")):
                            if key:
                                message = message.replace(key, "[redacted]")
                        result["message"] = message[:300]
                return result
            except (httpx.HTTPError, ValueError):
                return {"provider": name, "status": "connection_failed"}
        return await asyncio.gather(
            check("ElevenLabs", "https://api.elevenlabs.io/v1/convai/agents",
                  {"xi-api-key": os.getenv("ELEVENLABS_API_KEY", "")}),
            check("Gemini", "https://generativelanguage.googleapis.com/v1beta/models",
                  {"x-goog-api-key": os.getenv("GEMINI_API_KEY", "")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-gemini", action="store_true",
                        help="Generate one short explanation of synthetic data (uses API quota).")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(check_access()), indent=2))
    if args.test_gemini:
        reporter = GeminiReporter(os.getenv("GEMINI_API_KEY", ""), os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"))
        synthetic = {"sequence": 1, "task": "conversation", "features": {}, "units": {},
                     "quality": {"usable": False, "reasons": ["insufficient_voicing"]},
                     "classification": {"status": "insufficient_audio_quality", "score_pd": None,
                                        "interpretation": "No disease assessment is available."}}
        result = asyncio.run(reporter.generate(synthetic))
        print(json.dumps({k: v for k, v in result.items() if k != "evidence"}, indent=2))
