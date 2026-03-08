from __future__ import annotations

from typing import Protocol

import numpy as np
import pyttsx3
import sounddevice as sd

from voice_agent.config import Settings


class TTSClient(Protocol):
    def speak(self, text: str) -> None:
        ...


class PyttsxTTS:
    def __init__(self, rate: int = 180, voice_id: str = "") -> None:
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        if voice_id:
            self.engine.setProperty("voice", voice_id)

    def speak(self, text: str) -> None:
        if not text:
            return
        self.engine.say(text)
        self.engine.runAndWait()


class KokoroTTS:
    def __init__(self, lang_code: str = "a", voice: str = "af_heart", speed: float = 1.0) -> None:
        from kokoro import KPipeline

        self.pipeline = KPipeline(lang_code=lang_code)
        self.voice = voice
        self.speed = speed
        self.sample_rate = 24000

    def speak(self, text: str) -> None:
        if not text:
            return

        audio_chunks: list[np.ndarray] = []
        for _graphemes, _phonemes, audio in self.pipeline(
            text,
            voice=self.voice,
            speed=self.speed,
        ):
            audio_chunks.append(np.asarray(audio, dtype=np.float32))

        if not audio_chunks:
            return

        waveform = np.concatenate(audio_chunks, axis=0)
        sd.play(waveform, self.sample_rate)
        sd.wait()


def build_tts(settings: Settings) -> TTSClient:
    requested = settings.tts_backend
    if requested == "pyttsx3":
        print("[tts] Using pyttsx3 backend.")
        return PyttsxTTS(rate=settings.tts_rate, voice_id=settings.tts_voice_id)

    try:
        print("[tts] Loading Kokoro TTS (hexgrad/Kokoro-82M) from Hugging Face...")
        return KokoroTTS(
            lang_code=settings.kokoro_lang_code,
            voice=settings.kokoro_voice,
            speed=settings.kokoro_speed,
        )
    except Exception as exc:
        print(f"[tts] Kokoro unavailable, falling back to pyttsx3: {exc}")
        return PyttsxTTS(rate=settings.tts_rate, voice_id=settings.tts_voice_id)

