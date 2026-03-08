from __future__ import annotations

import queue
import re
import threading
from typing import Callable

import numpy as np

from voice_agent.auto_stream import SileroVADListener
from voice_agent.asr import WhisperTranscriber
from voice_agent.audio import ensure_input_device, record_audio_turn
from voice_agent.config import load_settings
from voice_agent.llm import LocalPhiClient
from voice_agent.session import ChatSession
from voice_agent.tts import build_tts


class StreamingSpeaker:
    def __init__(self, tts: object, mode: str = "sentence", min_words: int = 4) -> None:
        self._tts = tts
        self._mode = mode if mode in {"sentence", "chunk"} else "sentence"
        self._min_words = max(1, min_words)
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._buffer = ""
        self._buffer_word_count = 0
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                self._tts.speak(item)
            except Exception as exc:
                print(f"[tts] failed during streaming speak: {exc}")

    def _flush_buffer(self) -> None:
        text = self._buffer.strip()
        if text:
            self._queue.put(text)
        self._buffer = ""
        self._buffer_word_count = 0

    def feed(self, chunk: str) -> None:
        if not chunk:
            return

        self._buffer += chunk

        if self._mode == "sentence":
            while True:
                match = re.search(r"[.!?]+(?=\s|$)", self._buffer)
                if not match:
                    break
                sentence = self._buffer[: match.end()].strip()
                if sentence:
                    self._queue.put(sentence)
                self._buffer = self._buffer[match.end() :].lstrip()
            return

        self._buffer_word_count += len(re.findall(r"\b\w+\b", chunk))

        if any(mark in chunk for mark in ".!?"):
            self._flush_buffer()
            return

        if self._buffer_word_count >= self._min_words and self._buffer.endswith(" "):
            self._flush_buffer()

    def finish(self) -> None:
        self._flush_buffer()
        self._queue.put(None)
        self._thread.join()


def run_agent_turn(
    user_text: str,
    session: ChatSession,
    llm: LocalPhiClient,
    tts: object,
    tts_streaming: bool,
    tts_stream_mode: str,
    tts_stream_min_words: int,
    on_tts_start: Callable[[], None] | None = None,
    on_tts_end: Callable[[], None] | None = None,
) -> None:
    if not user_text:
        print("[asr] I could not hear clear speech. Try again.\n")
        return
    print(f"\nYou: {user_text}")

    session.add_user(user_text)
    speaker: StreamingSpeaker | None = None
    tts_locked = False
    try:
        print("Assistant: ", end="", flush=True)
        chunks: list[str] = []
        if tts_streaming:
            if on_tts_start:
                on_tts_start()
                tts_locked = True
            speaker = StreamingSpeaker(
                tts=tts,
                mode=tts_stream_mode,
                min_words=tts_stream_min_words,
            )
        for chunk in llm.chat_stream(session.messages()):
            print(chunk, end="", flush=True)
            chunks.append(chunk)
            if speaker:
                speaker.feed(chunk)
        print()
        reply = "".join(chunks).strip()
        if "</think>" in reply:
            reply = reply.split("</think>", 1)[-1].strip()
    except RuntimeError as exc:
        if speaker:
            speaker.finish()
        if tts_locked and on_tts_end:
            on_tts_end()
        print(f"[llm] {exc}\n")
        return

    if not reply:
        if speaker:
            speaker.finish()
        if tts_locked and on_tts_end:
            on_tts_end()
        print("[llm] Empty response from model.\n")
        return

    print()
    session.add_assistant(reply)
    try:
        if speaker:
            speaker.finish()
            if tts_locked and on_tts_end:
                on_tts_end()
        else:
            if on_tts_start:
                on_tts_start()
                tts_locked = True
            tts.speak(reply)
            if tts_locked and on_tts_end:
                on_tts_end()
    except Exception as exc:
        if tts_locked and on_tts_end:
            on_tts_end()
        print(f"[tts] failed to speak response: {exc}")


def run() -> int:
    try:
        settings = load_settings()
    except ValueError as exc:
        print(f"[config] {exc}")
        return 1

    try:
        ensure_input_device()
    except RuntimeError as exc:
        print(f"[audio] {exc}")
        return 1

    print("Loading local Whisper model. First run may take a while.")
    transcriber = WhisperTranscriber(
        model_name=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
    print(
        "Loading local LLM "
        f"({settings.llm_model_id}) with {settings.llm_precision} precision. "
        "First run may download weights."
    )
    try:
        llm = LocalPhiClient(
            model_id=settings.llm_model_id,
            hf_token=settings.hf_token,
            local_files_only=settings.hf_local_files_only,
            device=settings.llm_device,
            precision=settings.llm_precision,
            max_new_tokens=settings.llm_max_new_tokens,
            temperature=settings.llm_temperature,
            top_p=settings.llm_top_p,
        )
    except Exception as exc:
        print(f"[llm] failed to load local model: {exc}")
        return 1
    tts = build_tts(settings)
    session = ChatSession(
        system_prompt=settings.system_prompt,
        max_history_turns=settings.max_history_turns,
    )

    print("\nVoice agent ready.")
    if settings.auto_stream:
        print("Auto stream mode is ON.")
        print("Silero VAD will detect turn boundaries from your speech.")
        print("Use Ctrl+C to stop.\n")
    else:
        print("Press Enter to start recording a turn.")
        print("While recording, press Enter again to stop.")
        print("Type 'quit' or 'exit' to leave.\n")

    try:
        if settings.auto_stream:
            try:
                if settings.vad_provider != "silero":
                    raise RuntimeError(
                        f"Unsupported VAD_PROVIDER '{settings.vad_provider}'. Use 'silero'."
                    )
                listener = SileroVADListener(
                    sample_rate=settings.audio_sample_rate,
                    threshold=settings.vad_threshold,
                    min_speech_ms=settings.vad_min_speech_ms,
                    min_silence_ms=settings.vad_min_silence_ms,
                    pre_speech_ms=settings.vad_pre_speech_ms,
                    max_utterance_s=settings.auto_stream_max_utterance_s,
                )

                def on_tts_start() -> None:
                    listener.set_muted(True)

                def on_tts_end() -> None:
                    listener.clear_backlog()
                    listener.set_muted(False)

                for audio in listener.iter_utterances():
                    user_text = transcriber.transcribe(np.asarray(audio, dtype=np.float32))
                    run_agent_turn(
                        user_text=user_text,
                        session=session,
                        llm=llm,
                        tts=tts,
                        tts_streaming=settings.tts_streaming,
                        tts_stream_mode=settings.tts_stream_mode,
                        tts_stream_min_words=settings.tts_stream_min_words,
                        on_tts_start=on_tts_start,
                        on_tts_end=on_tts_end,
                    )
            except Exception as exc:
                print(f"[auto-stream] {exc}")
                return 1
        else:
            while True:
                action = input("Press Enter to talk (or type quit): ").strip().lower()
                if action in {"quit", "exit"}:
                    print("Bye.")
                    break
                if action:
                    print("Unknown command. Press Enter to record, or type quit.")
                    continue

                try:
                    audio = record_audio_turn(
                        sample_rate=settings.audio_sample_rate,
                        channels=settings.audio_channels,
                        max_seconds=settings.audio_max_seconds,
                    )
                except Exception as exc:
                    print(f"[audio] failed to record: {exc}")
                    continue

                user_text = transcriber.transcribe(audio)
                run_agent_turn(
                    user_text=user_text,
                    session=session,
                    llm=llm,
                    tts=tts,
                    tts_streaming=settings.tts_streaming,
                    tts_stream_mode=settings.tts_stream_mode,
                    tts_stream_min_words=settings.tts_stream_min_words,
                )
    except KeyboardInterrupt:
        print("\nInterrupted. Bye.")
    finally:
        llm.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(run())

