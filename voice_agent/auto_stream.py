from __future__ import annotations

from collections import deque
import queue
import threading

import numpy as np
import sounddevice as sd
import torch


class SileroVADListener:
    """Continuously listens to microphone and yields finalized utterances."""

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 700,
        pre_speech_ms: int = 200,
        max_utterance_s: int = 20,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_samples = int(sample_rate * min_speech_ms / 1000)
        self.min_silence_samples = int(sample_rate * min_silence_ms / 1000)
        self.pre_speech_samples = int(sample_rate * pre_speech_ms / 1000)
        self.max_utterance_samples = int(sample_rate * max_utterance_s)

        self.chunk_samples = 512
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.pre_roll: deque[np.ndarray] = deque(maxlen=max(1, self.pre_speech_samples // self.chunk_samples))
        self._mute_lock = threading.Lock()
        self._muted = False
        self._stop_event = threading.Event()

        # First run will download the Silero VAD model.
        self.model, _utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            onnx=False,
            trust_repo=True,
        )
        self.model.to("cpu")
        self.model.eval()

    def _speech_probability(self, chunk: np.ndarray) -> float:
        if chunk.size == 0:
            return 0.0
        if chunk.shape[0] != self.chunk_samples:
            padded = np.zeros(self.chunk_samples, dtype=np.float32)
            padded[: min(self.chunk_samples, chunk.shape[0])] = chunk[: self.chunk_samples]
            chunk = padded
        tensor = torch.from_numpy(chunk).unsqueeze(0)
        with torch.inference_mode():
            prob = self.model(tensor, self.sample_rate).item()
        return float(prob)

    def set_muted(self, muted: bool) -> None:
        with self._mute_lock:
            self._muted = muted

    def clear_backlog(self) -> None:
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        self.pre_roll.clear()

    def stop(self) -> None:
        self._stop_event.set()

    def iter_utterances(self):
        in_speech = False
        speech_samples = 0
        silence_samples = 0
        speech_frames: list[np.ndarray] = []

        def on_audio(indata: np.ndarray, _frames: int, _time: object, status: object) -> None:
            if status:
                print(f"[audio] input status: {status}")
            with self._mute_lock:
                if self._muted:
                    return
            mono = np.squeeze(indata).astype(np.float32)
            if mono.ndim > 1:
                mono = mono.mean(axis=1).astype(np.float32)
            self.audio_queue.put(mono)

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_samples,
            callback=on_audio,
        ):
            print("[auto-stream] listening... speak naturally, pause to end turn.")

            while not self._stop_event.is_set():
                try:
                    chunk = self.audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                prob = self._speech_probability(chunk)
                is_speech = prob >= self.threshold
                self.pre_roll.append(chunk.copy())

                if not in_speech:
                    if is_speech:
                        in_speech = True
                        speech_frames = list(self.pre_roll)
                        speech_samples = sum(len(frame) for frame in speech_frames)
                        silence_samples = 0
                    continue

                speech_frames.append(chunk)
                speech_samples += len(chunk)

                if is_speech:
                    silence_samples = 0
                else:
                    silence_samples += len(chunk)

                reached_max = speech_samples >= self.max_utterance_samples
                reached_pause = silence_samples >= self.min_silence_samples
                enough_speech = speech_samples >= self.min_speech_samples

                if reached_max or (reached_pause and enough_speech):
                    utterance = np.concatenate(speech_frames, axis=0).astype(np.float32)
                    in_speech = False
                    speech_samples = 0
                    silence_samples = 0
                    speech_frames = []
                    self.pre_roll.clear()

                    if utterance.size > 0:
                        yield utterance

