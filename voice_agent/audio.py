from __future__ import annotations

import threading
import time

import numpy as np
import sounddevice as sd


def ensure_input_device() -> None:
    try:
        devices = sd.query_devices()
    except Exception as exc:
        raise RuntimeError(f"Could not query audio devices: {exc}") from exc

    has_input = any((device.get("max_input_channels", 0) or 0) > 0 for device in devices)
    if not has_input:
        raise RuntimeError("No audio input device detected. Connect a microphone and try again.")


def record_audio_turn(
    sample_rate: int,
    channels: int = 1,
    max_seconds: int = 20,
) -> np.ndarray:
    frames: list[np.ndarray] = []
    stop_event = threading.Event()

    def on_audio(indata: np.ndarray, _frame_count: int, _time: object, status: object) -> None:
        if status:
            print(f"[audio] input status: {status}")
        frames.append(indata.copy())

    def wait_for_stop_signal() -> None:
        input()
        stop_event.set()

    print("Recording... press Enter to stop.")
    print(f"Auto-stop in {max_seconds}s if you do not press Enter.")

    stopper = threading.Thread(target=wait_for_stop_signal, daemon=True)
    stopper.start()

    start = time.monotonic()
    with sd.InputStream(
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        callback=on_audio,
    ):
        while not stop_event.is_set():
            elapsed = time.monotonic() - start
            if elapsed >= max_seconds:
                print("[audio] reached max recording duration, stopping.")
                break
            sd.sleep(100)

    if not frames:
        return np.array([], dtype=np.float32)

    audio = np.concatenate(frames, axis=0)
    if channels > 1:
        # Convert multi-channel input to mono for ASR by averaging channels.
        audio = audio.mean(axis=1, keepdims=True)
    return np.squeeze(audio).astype(np.float32)

