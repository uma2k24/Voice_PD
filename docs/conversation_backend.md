# Conversational Parkinson's voice classifier

This branch adds a backend for the original proposal: a voice conversation with
an ElevenLabs Agent, live Parselmouth measurements, and a CPU acoustic classifier.
The inherited Flask app remains available as the earlier recording-based demo.
Use `conversation_app.py` for the new backend; no frontend changes are required.

The score is a research classifier output, not a diagnosis or a calibrated
personal probability of having Parkinson's. Scores are withheld for users under
50, poor-quality audio, missing models, and tasks absent from the training data.

## Run on Windows

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_conversation.txt
$env:ELEVENLABS_API_KEY = "your-key"
$env:ELEVENLABS_AGENT_ID = "your-agent-id"
.\.venv\Scripts\python.exe -m uvicorn conversation_app:app --host 127.0.0.1 --port 8000
```

Environment variables are documented in `.env.example`. The example file is not
automatically loaded; copy it to `.env` for automatic local loading. Existing
environment variables take priority over `.env`. Never commit the real `.env`.
The default model directory is `models_conversation`;
override it with `PD_MODEL_DIR`. Model files must come from trusted local training.
Restart the service after training to load the new model.

In a second terminal, use the microphone client (headphones recommended):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements_microphone.txt
.\.venv\Scripts\python.exe -m scripts.conversation_cli --age 60
```

The reference client suppresses microphone capture during agent playback. It is
half duplex. A future frontend can support interruption using echo cancellation;
it must keep agent playback out of the analysis audio and send `reset_audio` after
capture gaps. Audio is not saved by the conversation backend. Microphone audio is
sent to ElevenLabs; configure retention in your ElevenLabs account separately.

## Configure the ElevenLabs agent

Create an ElevenLabs Agent with a voice and a supported conversation LLM.
Alternatively, preview or create the dedicated agent using:

```powershell
.\.venv\Scripts\python.exe -m scripts.configure_elevenlabs_agent
.\.venv\Scripts\python.exe -m scripts.configure_elevenlabs_agent --create
```

The first command writes a credential-free configuration preview to `.local`.
The second creates a private agent with both client tools, PCM audio and prompt
overrides, then stores its ID in `.env`. A configured dedicated agent is reused.
Creation needs Agents read/write access; voice-library read access is not needed
for the standard default voice. `--voice-id` can select another permitted voice.

Configure input and output audio to **pcm_16000** for this reference client.
Enable overrides for the system prompt and first message. The server supplies
the research prompt from `src/elevenlabs_agent.py`. Enable client events for
`audio`, `interruption`, `agent_response`, `user_transcript`, and `client_tool_call`.
Metadata and ping events are handled by the server. API keys and signed URLs
remain on the backend.

Add two client tools in the agent dashboard, both with **Wait for response**:

| Tool | Parameters | Purpose |
| --- | --- | --- |
| `set_recording_task` | Required string `task`, enum `conversation`, `reading`, `sustained_vowel` | Switch the actual speech exercise and clear mixed-task windows. |
| `get_voice_analysis` | None | Fetch the latest completed measurements/status; returns `collecting` before completion. |

The configured agent invites the user to a comfortable sustained "ah" exercise
when the available classifier covers vowels. It calls `set_recording_task` before
the exercise, waits for the tool result, and fetches the analysis afterwards.
Ordinary conversation is never labelled as a sustained vowel. Training on reading
does not validate spontaneous conversation, and training on Italian speech does
not establish language-independent performance.

Protocol references: [ElevenLabs Agent WebSockets](https://elevenlabs.io/docs/eleven-agents/api-reference/eleven-agents/websocket),
[client tools](https://elevenlabs.io/docs/eleven-agents/customization/tools/client-tools),
and [signed URLs](https://elevenlabs.io/docs/api-reference/conversations/get-signed-url).

## Acoustic measurements

| JSON feature | Unit | Method |
| --- | --- | --- |
| `jitter_local_percent` | % | Praat local jitter; fraction multiplied by 100. |
| `shimmer_local_percent` | % | Praat local shimmer; fraction multiplied by 100. |
| `hnr_db` | dB | Praat cross-correlation harmonicity mean. |
| `cpps_db` | dB | Smoothed Praat PowerCepstrogram CPPS with 10 ms time and 1 ms quefrency averaging. |
| `gne_ratio` | ratio | Maximum of Praat's GNE band-envelope correlation matrix; 500–4500 Hz centres, 1000 Hz bandwidth, 80 Hz step. |
| `f0_mean_hz`, `f0_std_hz` | Hz | Mean and standard deviation over voiced pitch frames. |
| `f1_mean_hz`, `f2_mean_hz`, `f3_mean_hz` | Hz | Burg formant means at voiced-frame times. |
| `mean_period_seconds` | s | Mean accepted glottal pulse period. This is cycle duration, not regularity. |

Train and serve use the same extractor/version, sample rate, parameters and
four-second windows. The audio is resampled to 16 kHz, without amplitude
normalisation or dynamic compression. Quality checks reject low RMS, excessive
clipping, insufficient voicing and missing core measurements. Missing values
appear as JSON `null`; no NaN/Infinity values are sent. The fixed quality checks
are engineering heuristics and require cohort/device validation.

Windows are scheduled every second after the first four seconds of captured
audio. Computation runs off the WebSocket event loop. Each session holds only
one pending window and drops older pending work if processing is slower than
capture. `processing_ms` exposes actual latency. GNE is relatively expensive;
one-second scheduling does not guarantee one result per second. Window times
refer to captured microphone samples, excluding playback/capture gaps.

See [Praat CPPS](https://praat.org/manual/PowerCepstrogram__Get_CPPS___.html)
and [Praat GNE implementation](https://github.com/praat/praat/blob/master/fon/Sound_to_Harmonicity_GNE.cpp)
for the underlying methods. The legacy `CPP` feature is a different computation
and is not silently substituted for CPPS.

## Train the original-proposal baseline

Supply a CSV manifest with these columns:

```csv
path,speaker_id,label,age,sex,task
audio/s01_a.wav,italian:s01,0,65,F,sustained_vowel
audio/s02_a.wav,italian:s02,1,67,M,sustained_vowel
```

Paths are relative to the manifest. Labels must be HC=0 and PD=1. IDs must
identify the same person across all recordings, sessions and augmentations.
Age, sex, label and task must be supplied from reliable metadata. Age and sex
are cohort metadata, not classifier input features.

For the dataset already in this checkout:

```powershell
.\.venv\Scripts\python.exe -m scripts.build_ipvs_manifest --task sustained_vowel
.\.venv\Scripts\python.exe -m scripts.train_conversation_classifier --manifest data/ipvs_manifest.csv
```

The manifest builder joins original workbook ages/sex to speaker folders with
exact normalised name/surname matches. Ambiguous and unmatched records are
excluded and written to `data/ipvs_manifest.audit.json`. It uses only the /a/
task codes VA1/VA2, or B1/B2 when `--task reading` is supplied. It never decodes
age from filenames. Manual correction of unmatched metadata requires independent
evidence. The included model's `metadata.json` reports the usable training cohort.

The initial local Logistic Regression run used 186 usable windows from 36
speakers and produced a held-out speaker AUC of 0.876 and balanced accuracy of
0.777 at threshold 0.5. This is within-dataset, sustained-/a/ evaluation; no
spontaneous-conversation or MDVR-KCL external evaluation has been performed.

The training command:

- Filters to age >=50 before feature extraction.
- Uniformly samples up to three four-second windows per recording by default.
- Fits median imputation, scaling and fixed-hyperparameter Logistic Regression
  inside speaker-separated StratifiedGroupKFold folds.
- Supports `--classifier random_forest` and `--loso` as explicit alternatives.
- Reports held-out **speaker-level** AUC, balanced accuracy at fixed threshold
  0.5, and Brier score. No selection/tuning uses the reported held-out scores.
- Refits a production pipeline on all eligible usable data and writes
  `classifier.joblib`, `metadata.json`, fold membership, per-speaker held-out scores,
  accepted training windows and rejected-window reasons.

The default five folds need at least five speakers per class after quality
filtering. Generated manifests and detailed participant-level outputs are ignored
by Git. The trained classifier and aggregate metadata are retained.

For external validation, supply `--external-manifest path/to/mdvr_manifest.csv`.
It checks speaker and recording disjointness and task compatibility, and reports
external speaker-level metrics without refitting. Use matched reading tasks for
Italian-to-MDVR reading evaluation; a vowel model cannot be tested as a reading
model. Recording IDs alone cannot detect renamed copies or incorrect speaker IDs;
the manifest owner remains responsible for globally correct identities.

This classifier-focused branch implements the acoustic baseline and Gemini
explanations. openSMILE and Whisper/HuBERT/WavLM fusion remain subsequent stages
of the proposal; they are not claimed as implemented or evaluated here. The
inherited classifiers and their performance claims are not used by this backend.

## Gemini reports

Set `GEMINI_API_KEY` and optionally `GEMINI_MODEL` in `.env`. The verified default
is `gemini-3.1-flash-lite`. Reports contain authoritative server evidence plus
a short generated explanation. The model receives only allowlisted measurements,
units, quality, task coverage and classifier status; no raw audio or conversation
transcripts are sent to Gemini. It is instructed to avoid diagnoses, invented
thresholds, feature-importance claims and personal disease probabilities.

Send `{"type":"report"}` during a session to explain a completed snapshot.
`end` also includes a report in the final summary. Repeated requests for the
same analysis reuse the report. The report's `analysis_sequence` identifies its
source snapshot; a newer window may exist by the time generation completes.
Switching tasks invalidates reports still being generated for the previous task.
Provider failures leave the measured/classified results intact and return an
explicit report status. Explanations are generated research text and should be
reviewed before clinical use.

Gemini uses the documented [generateContent API](https://ai.google.dev/api/generate-content).
Successful model discovery alone does not prove a model is enabled for generation;
the connectivity helper can test a synthetic explanation:

```powershell
.\.venv\Scripts\python.exe -m scripts.check_provider_access
.\.venv\Scripts\python.exe -m scripts.check_provider_access --test-gemini
.\.venv\Scripts\python.exe -m scripts.verify_live_conversation
```

The last two commands use provider quota. The live conversation check verifies
the actual ElevenLabs handshake and greeting without recording a person.

## WebSocket contract

Connect to `ws://127.0.0.1:8000/api/conversation`. First send:

```json
{"type":"start","age":60,"task":"conversation"}
```

Wait for `ready`. Send little-endian mono signed 16-bit PCM at 16 kHz in binary
messages of at most one second (100 ms recommended). JSON audio is also accepted:
`{"type":"audio","audio":"BASE64_PCM"}`. Supported controls are `set_task`
with a `task`, `reset_audio`, `report`, and `end`. `set_task` must reflect the actual speech
exercise and resets the measurement window. Server-side agent tools can manage
this automatically. Finish with `{"type":"end"}` to receive a summary.

Server events include ElevenLabs audio/transcript/interruption events,
`task_changed`, `analysis` (features, units, quality, classification and timing),
`report`, `summary`, and `error`. `classification.score_pd` is null unless scoring is
eligible. Reasons include `outside_evaluated_population`,
`insufficient_audio_quality`, `unsupported_recording_task`, and `unavailable`.
Capture-gap resets preserve completed clean analyses and let clean in-flight
windows finish. Task switches invalidate analyses from the previous task.
The summary is the latest analysis of the current task, not an
aggregate disease assessment.

`GET /health` reports provider/model configuration. Set `PD_API_TOKEN` for
nonlocal access; clients supply an `Authorization: Bearer ...` header.
`PD_ALLOWED_ORIGINS` controls browser origins. The service does not create users
or store conversation audio. Idle connections time out after 45 seconds and
sessions are limited to ten minutes.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover actual Praat measurements, poor audio, bounded windows, cohort/task
gating, artifact round trips, speaker-disjoint folds and the streaming protocol
with a fake ElevenLabs transport, Gemini failure handling, evidence allowlisting
and report reuse. A full microphone conversation needs an API key, configured
agent, microphone and speakers. Unit tests never use the real keys in `.env`.
