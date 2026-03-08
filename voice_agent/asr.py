from __future__ import annotations

from faster_whisper import WhisperModel
import numpy as np


class WhisperTranscriber:
    def __init__(self, model_name: str, device: str = "auto", compute_type: str = "int8") -> None:
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""

        segments, _ = self.model.transcribe(audio, vad_filter=True, beam_size=1)
        text = "".join(segment.text for segment in segments).strip()
        return text

