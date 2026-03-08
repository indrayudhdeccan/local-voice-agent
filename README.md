# Voice Agent (Local ASR + Local LLM + Local TTS)

Production-oriented local voice assistant with:
- ASR via `faster-whisper`
- LLM via local `transformers` model loading
- TTS via Kokoro (`hexgrad/Kokoro-82M`) with `pyttsx3` fallback
- Always-listening mode with Silero VAD
- CLI and desktop UI entrypoints

## Features

- **Local-first pipeline**: no cloud dependency required for core voice loop
- **Two interfaces**:
  - CLI (`main.py`) for terminal-centric workflows
  - Desktop UI (`ui_app.py`) for model/runtime selection and interactive use
- **Streaming controls**:
  - LLM streaming on/off
  - TTS stream modes (`sentence`, `chunk`, `none`)
- **Configurable runtime** via `.env` with sane defaults
- **Offline model mode** (`HF_LOCAL_FILES_ONLY=true`) to avoid remote checks

## Architecture

```text
Microphone -> VAD (Silero) -> ASR (Whisper) -> LLM (Transformers) -> TTS (Kokoro/pyttsx3)
```

Core modules:
- `voice_agent/config.py` environment loading + validation
- `voice_agent/auto_stream.py` always-on VAD listener
- `voice_agent/asr.py` Whisper transcription
- `voice_agent/llm.py` local model inference (streaming + non-streaming)
- `voice_agent/tts.py` speech synthesis abstraction

## Prerequisites

- Python 3.10+
- Working microphone and speaker output
- macOS dependencies:
  - `brew install portaudio`
  - `brew install espeak-ng`
- Enough RAM/VRAM for selected LLM

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Run CLI:

```bash
python3 main.py
```

Run desktop UI:

```bash
python3 ui_app.py
```

## Configuration

Copy `.env.example` and tune as needed.

Important keys:
- `LLM_MODEL_ID`: model to load with `transformers`
- `HF_LOCAL_FILES_ONLY`: load strictly from local cache
- `AUTO_STREAM`: always-listening mode
- `VAD_*`: speech endpointing behavior
- `TTS_BACKEND`: `kokoro` or `pyttsx3`
- `TTS_STREAM_MODE`: `sentence`, `chunk`, or `none`

Notes:
- For public models (e.g. `microsoft/Phi-4-mini-instruct`), `HUGGINGFACE_HUB_TOKEN` is optional.
- For gated/private models, token is required.

## UI Capabilities

`ui_app.py` includes:
- Stack dropdowns for VAD/ASR/LLM/TTS
- LLM streaming toggle
- TTS stream mode selector
- Stage-based load progress bar
- Hidden hyperparameter drawer (`---`) with context-aware fields
- Always-listening control (`Start/Stop Listening`)

## Troubleshooting

- **Slow startup**
  - Initial model download can be large.
  - Subsequent starts still load shards into memory; keep process running for best UX.
- **Model load failure**
  - Verify `LLM_MODEL_ID` exists and is compatible.
  - Set `HF_LOCAL_FILES_ONLY=false` if cache is missing.
- **No mic input**
  - Check OS mic permissions and selected input device.
- **TTS feedback loop**
  - Always-listening path mutes and clears backlog during TTS, but keep speaker volume reasonable.

