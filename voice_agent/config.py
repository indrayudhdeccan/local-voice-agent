from __future__ import annotations

import os
from dataclasses import dataclass, replace

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    llm_model_id: str
    hf_token: str
    hf_local_files_only: bool
    llm_device: str
    llm_precision: str
    llm_max_new_tokens: int
    llm_temperature: float
    llm_top_p: float
    system_prompt: str
    max_history_turns: int
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    auto_stream: bool
    vad_provider: str
    vad_threshold: float
    vad_min_speech_ms: int
    vad_min_silence_ms: int
    vad_pre_speech_ms: int
    auto_stream_max_utterance_s: int
    audio_sample_rate: int
    audio_channels: int
    audio_max_seconds: int
    tts_backend: str
    tts_rate: int
    tts_voice_id: str
    tts_streaming: bool
    tts_stream_mode: str
    tts_stream_min_words: int
    kokoro_lang_code: str
    kokoro_voice: str
    kokoro_speed: float


def _env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name, "true" if default else "false").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def load_settings() -> Settings:
    load_dotenv()

    settings = Settings(
        llm_model_id=_env("LLM_MODEL_ID", "microsoft/Phi-4-mini-instruct"),
        hf_token=_env("HUGGINGFACE_HUB_TOKEN", ""),
        hf_local_files_only=_env_bool("HF_LOCAL_FILES_ONLY", True),
        llm_device=_env("LLM_DEVICE", "auto"),
        llm_precision=_env("LLM_PRECISION", "float16").lower(),
        llm_max_new_tokens=int(_env("LLM_MAX_NEW_TOKENS", "120")),
        llm_temperature=float(_env("LLM_TEMPERATURE", "0.4")),
        llm_top_p=float(_env("LLM_TOP_P", "0.9")),
        system_prompt=_env(
            "SYSTEM_PROMPT",
            (
                "You are a concise, conversational voice assistant. "
                "Reply in 1-3 short sentences unless the user asks for detail. "
                "Do not include step-by-step reasoning unless explicitly requested."
            ),
        ),
        max_history_turns=int(_env("MAX_HISTORY_TURNS", "6")),
        whisper_model=_env("WHISPER_MODEL", "base"),
        whisper_device=_env("WHISPER_DEVICE", "auto"),
        whisper_compute_type=_env("WHISPER_COMPUTE_TYPE", "int8"),
        auto_stream=_env_bool("AUTO_STREAM", False),
        vad_provider=_env("VAD_PROVIDER", "silero").strip().lower(),
        vad_threshold=float(_env("VAD_THRESHOLD", "0.5")),
        vad_min_speech_ms=int(_env("VAD_MIN_SPEECH_MS", "250")),
        vad_min_silence_ms=int(_env("VAD_MIN_SILENCE_MS", "700")),
        vad_pre_speech_ms=int(_env("VAD_PRE_SPEECH_MS", "200")),
        auto_stream_max_utterance_s=int(_env("AUTO_STREAM_MAX_UTTERANCE_S", "20")),
        audio_sample_rate=int(_env("AUDIO_SAMPLE_RATE", "16000")),
        audio_channels=int(_env("AUDIO_CHANNELS", "1")),
        audio_max_seconds=int(_env("AUDIO_MAX_SECONDS", "20")),
        tts_backend=_env("TTS_BACKEND", "kokoro").lower(),
        tts_rate=int(_env("TTS_RATE", "180")),
        tts_voice_id=_env("TTS_VOICE_ID", ""),
        tts_streaming=_env_bool("TTS_STREAMING", True),
        tts_stream_mode=_env("TTS_STREAM_MODE", "sentence").strip().lower(),
        tts_stream_min_words=int(_env("TTS_STREAM_MIN_WORDS", "4")),
        kokoro_lang_code=_env("KOKORO_LANG_CODE", "a"),
        kokoro_voice=_env("KOKORO_VOICE", "af_heart"),
        kokoro_speed=float(_env("KOKORO_SPEED", "1.0")),
    )
    if settings.tts_stream_mode == "none" and settings.tts_streaming:
        settings = replace(settings, tts_streaming=False)

    _validate_settings(settings)
    return settings


def _validate_settings(settings: Settings) -> None:
    if settings.llm_max_new_tokens <= 0:
        raise ValueError("LLM_MAX_NEW_TOKENS must be > 0.")
    if not (0.0 <= settings.llm_temperature <= 2.0):
        raise ValueError("LLM_TEMPERATURE must be in range [0.0, 2.0].")
    if not (0.0 < settings.llm_top_p <= 1.0):
        raise ValueError("LLM_TOP_P must be in range (0.0, 1.0].")
    if settings.max_history_turns <= 0:
        raise ValueError("MAX_HISTORY_TURNS must be > 0.")
    if settings.audio_sample_rate <= 0:
        raise ValueError("AUDIO_SAMPLE_RATE must be > 0.")
    if settings.audio_channels <= 0:
        raise ValueError("AUDIO_CHANNELS must be > 0.")
    if settings.audio_max_seconds <= 0:
        raise ValueError("AUDIO_MAX_SECONDS must be > 0.")
    if settings.vad_provider not in {"silero"}:
        raise ValueError("VAD_PROVIDER must currently be 'silero'.")
    if not (0.0 <= settings.vad_threshold <= 1.0):
        raise ValueError("VAD_THRESHOLD must be in range [0.0, 1.0].")
    if settings.vad_min_speech_ms <= 0:
        raise ValueError("VAD_MIN_SPEECH_MS must be > 0.")
    if settings.vad_min_silence_ms <= 0:
        raise ValueError("VAD_MIN_SILENCE_MS must be > 0.")
    if settings.vad_pre_speech_ms < 0:
        raise ValueError("VAD_PRE_SPEECH_MS must be >= 0.")
    if settings.auto_stream_max_utterance_s <= 0:
        raise ValueError("AUTO_STREAM_MAX_UTTERANCE_S must be > 0.")
    if settings.tts_backend not in {"kokoro", "pyttsx3"}:
        raise ValueError("TTS_BACKEND must be either 'kokoro' or 'pyttsx3'.")
    if settings.tts_rate <= 0:
        raise ValueError("TTS_RATE must be > 0.")
    if settings.tts_stream_mode not in {"sentence", "chunk", "none"}:
        raise ValueError("TTS_STREAM_MODE must be one of: sentence, chunk, none.")
    if settings.tts_stream_min_words <= 0:
        raise ValueError("TTS_STREAM_MIN_WORDS must be > 0.")
    if settings.kokoro_speed <= 0:
        raise ValueError("KOKORO_SPEED must be > 0.")

