from __future__ import annotations

import threading
import tkinter as tk
from tkinter import scrolledtext, ttk
from dataclasses import replace
from typing import Callable

import numpy as np

from voice_agent.auto_stream import SileroVADListener
from voice_agent.asr import WhisperTranscriber
from voice_agent.config import Settings, load_settings
from voice_agent.llm import LocalPhiClient
from voice_agent.session import ChatSession
from voice_agent.tts import build_tts


class VoiceAgentUI:
    ASR_OPTIONS = ["tiny", "base", "small", "medium", "large-v3"]
    LLM_OPTIONS = [
        "microsoft/Phi-4-mini-instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen3.5-4B",
    ]
    VAD_OPTIONS = ["silero"]
    TTS_OPTIONS = ["kokoro", "pyttsx3"]
    STREAM_MODE_OPTIONS = ["sentence", "chunk", "none"]

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Voice Agent Studio")
        self.root.geometry("980x700")

        self.settings: Settings | None = None
        self.transcriber: WhisperTranscriber | None = None
        self.llm: LocalPhiClient | None = None
        self.tts: object | None = None
        self.session: ChatSession | None = None

        self.is_busy = False
        self.is_processing_turn = False
        self.auto_listening = False
        self.listener: SileroVADListener | None = None

        self.vad_var = tk.StringVar(value="silero")
        self.asr_var = tk.StringVar(value="base")
        self.llm_var = tk.StringVar(value="microsoft/Phi-4-mini-instruct")
        self.tts_var = tk.StringVar(value="kokoro")
        self.stream_mode_var = tk.StringVar(value="sentence")
        self.llm_streaming_var = tk.BooleanVar(value=True)
        self.hp_visible = False
        self.hp_vars: dict[str, tk.Variable] = {}
        self.hp_rows: dict[str, tk.Widget] = {}

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._set_status("Ready. Pick models and click 'Load / Reload'.")
        self._set_indicator("idle")

    def _build_ui(self) -> None:
        top = tk.Frame(self.root, padx=10, pady=10)
        top.pack(fill=tk.X)

        status_row = tk.Frame(top)
        status_row.pack(fill=tk.X, pady=(0, 8))

        self.state_badge = tk.Label(status_row, text="IDLE", width=12, relief=tk.GROOVE)
        self.state_badge.pack(side=tk.LEFT, padx=(0, 8))

        self.status_label = tk.Label(status_row, text="", anchor="w")
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.menu_btn = tk.Button(status_row, text="---", width=4, command=self._toggle_hyperparams)
        self.menu_btn.pack(side=tk.RIGHT, padx=(8, 8))

        self.progress = ttk.Progressbar(status_row, mode="determinate", length=180, maximum=4)
        self.progress.pack(side=tk.RIGHT)

        options_frame = tk.LabelFrame(top, text="Runtime Stack", padx=8, pady=8)
        options_frame.pack(fill=tk.X, pady=(0, 8))

        self._add_labeled_combo(
            options_frame,
            "VAD",
            self.vad_var,
            self.VAD_OPTIONS,
            row=0,
            col=0,
        )
        self._add_labeled_combo(
            options_frame,
            "ASR (Whisper)",
            self.asr_var,
            self.ASR_OPTIONS,
            row=0,
            col=1,
        )
        self._add_labeled_combo(
            options_frame,
            "LLM",
            self.llm_var,
            self.LLM_OPTIONS,
            row=0,
            col=2,
        )
        self._add_labeled_combo(
            options_frame,
            "TTS",
            self.tts_var,
            self.TTS_OPTIONS,
            row=1,
            col=0,
        )
        self._add_labeled_combo(
            options_frame,
            "TTS Stream Mode",
            self.stream_mode_var,
            self.STREAM_MODE_OPTIONS,
            row=1,
            col=1,
        )

        llm_stream_container = tk.Frame(options_frame)
        llm_stream_container.grid(row=1, column=2, padx=8, pady=4, sticky="w")
        tk.Label(llm_stream_container, text="LLM Streaming").pack(anchor="w")
        tk.Checkbutton(
            llm_stream_container,
            text="Enable token streaming",
            variable=self.llm_streaming_var,
            onvalue=True,
            offvalue=False,
        ).pack(anchor="w")

        controls = tk.Frame(top)
        controls.pack(fill=tk.X)

        self.load_btn = tk.Button(controls, text="Load / Reload", command=self._load_models)
        self.load_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.listen_btn = tk.Button(
            controls,
            text="Start Listening",
            command=self._toggle_listening,
            state=tk.DISABLED,
        )
        self.listen_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.send_btn = tk.Button(
            controls,
            text="Send Text",
            command=self._send_text,
            state=tk.DISABLED,
        )
        self.send_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.clear_btn = tk.Button(
            controls,
            text="Clear Chat",
            command=self._clear_chat,
        )
        self.clear_btn.pack(side=tk.LEFT)

        input_row = tk.Frame(self.root, padx=10)
        input_row.pack(fill=tk.X)

        self.text_input = tk.Entry(input_row)
        self.text_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.text_input.bind("<Return>", lambda _event: self._send_text())

        body = tk.Frame(self.root, padx=10, pady=10)
        body.pack(fill=tk.BOTH, expand=True)

        self.left_body = tk.Frame(body)
        self.left_body.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.chat = scrolledtext.ScrolledText(self.left_body, wrap=tk.WORD, state=tk.DISABLED)
        self.chat.pack(fill=tk.BOTH, expand=True)

        self.hp_frame = tk.LabelFrame(body, text="Hyperparameters", padx=8, pady=8, width=320)
        self.hp_frame.pack_propagate(False)

        self.hp_canvas = tk.Canvas(self.hp_frame, highlightthickness=0)
        self.hp_scroll = ttk.Scrollbar(self.hp_frame, orient="vertical", command=self.hp_canvas.yview)
        self.hp_inner = tk.Frame(self.hp_canvas)
        self.hp_inner.bind(
            "<Configure>",
            lambda _e: self.hp_canvas.configure(scrollregion=self.hp_canvas.bbox("all")),
        )
        self.hp_canvas.create_window((0, 0), window=self.hp_inner, anchor="nw")
        self.hp_canvas.configure(yscrollcommand=self.hp_scroll.set)
        self.hp_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.hp_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._build_hyperparam_panel()
        self.vad_var.trace_add("write", lambda *_: self._refresh_hyperparam_visibility())
        self.tts_var.trace_add("write", lambda *_: self._refresh_hyperparam_visibility())
        self.stream_mode_var.trace_add("write", lambda *_: self._refresh_hyperparam_visibility())

    def _add_labeled_combo(
        self,
        parent: tk.Widget,
        label: str,
        var: tk.StringVar,
        values: list[str],
        row: int,
        col: int,
    ) -> None:
        container = tk.Frame(parent)
        container.grid(row=row, column=col, padx=8, pady=4, sticky="w")
        tk.Label(container, text=label).pack(anchor="w")
        combo = ttk.Combobox(container, textvariable=var, values=values, state="readonly", width=30)
        combo.pack(anchor="w")

    def _set_status(self, text: str) -> None:
        self.status_label.config(text=text)

    def _set_indicator(self, state: str) -> None:
        palette = {
            "idle": ("IDLE", "#d9d9d9"),
            "loading": ("LOADING", "#f4d35e"),
            "listening": ("LISTENING", "#7bd389"),
            "thinking": ("THINKING", "#89c2ff"),
            "speaking": ("SPEAKING", "#ffaf87"),
            "error": ("ERROR", "#ff6b6b"),
        }
        text, bg = palette.get(state, ("STATE", "#d9d9d9"))
        self.state_badge.config(text=text, bg=bg)

    def _log(self, text: str) -> None:
        self._append_chat(f"[system] {text}\n")

    def _append_chat(self, text: str) -> None:
        self.chat.configure(state=tk.NORMAL)
        self.chat.insert(tk.END, text)
        self.chat.see(tk.END)
        self.chat.configure(state=tk.DISABLED)

    def _clear_chat(self) -> None:
        self.chat.configure(state=tk.NORMAL)
        self.chat.delete("1.0", tk.END)
        self.chat.configure(state=tk.DISABLED)

    def _set_busy(self, busy: bool) -> None:
        self.is_busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self.load_btn.config(state=state)
        self.clear_btn.config(state=state)
        if busy:
            self.progress.configure(value=0)
            self._set_indicator("loading")
        else:
            self.progress.configure(value=0)
            if self.auto_listening:
                self._set_indicator("listening")
            else:
                self._set_indicator("idle")

        if self.llm is not None:
            self.listen_btn.config(state=tk.NORMAL if not busy else tk.DISABLED)
            self.send_btn.config(state=tk.NORMAL if not busy else tk.DISABLED)

    def _set_load_progress(self, step: int, message: str) -> None:
        self.progress.configure(value=max(0, min(4, step)))
        self._set_status(message)
        self._log(message)

    def _int_from_var(self, key: str, fallback: int) -> int:
        raw = str(self.hp_vars[key].get()).strip()
        try:
            return int(raw)
        except ValueError:
            return fallback

    def _float_from_var(self, key: str, fallback: float) -> float:
        raw = str(self.hp_vars[key].get()).strip()
        try:
            return float(raw)
        except ValueError:
            return fallback

    def _toggle_hyperparams(self) -> None:
        if self.hp_visible:
            self.hp_frame.pack_forget()
            self.hp_visible = False
        else:
            self.hp_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
            self.hp_visible = True

    def _register_hp_row(
        self,
        key: str,
        label: str,
        var: tk.Variable,
        widget_builder: Callable[[tk.Frame], tk.Widget],
    ) -> None:
        row = tk.Frame(self.hp_inner)
        row.pack(fill=tk.X, pady=3)
        tk.Label(row, text=label).pack(anchor="w")
        self.hp_vars[key] = var
        widget_builder(row)
        self.hp_rows[key] = row

    def _build_hyperparam_panel(self) -> None:
        self._register_hp_row(
            "hf_local_files_only",
            "HF Local Files Only",
            tk.BooleanVar(value=True),
            lambda row: tk.Checkbutton(row, variable=self.hp_vars["hf_local_files_only"]).pack(anchor="w"),
        )
        self._register_hp_row(
            "auto_stream",
            "Auto Stream",
            tk.BooleanVar(value=True),
            lambda row: tk.Checkbutton(row, variable=self.hp_vars["auto_stream"], command=self._refresh_hyperparam_visibility).pack(anchor="w"),
        )
        self._register_hp_row(
            "whisper_device",
            "Whisper Device",
            tk.StringVar(value="auto"),
            lambda row: tk.Entry(row, textvariable=self.hp_vars["whisper_device"]).pack(fill=tk.X),
        )
        self._register_hp_row(
            "whisper_compute_type",
            "Whisper Compute Type",
            tk.StringVar(value="int8"),
            lambda row: tk.Entry(row, textvariable=self.hp_vars["whisper_compute_type"]).pack(fill=tk.X),
        )
        self._register_hp_row(
            "llm_device",
            "LLM Device",
            tk.StringVar(value="auto"),
            lambda row: tk.Entry(row, textvariable=self.hp_vars["llm_device"]).pack(fill=tk.X),
        )
        self._register_hp_row(
            "llm_precision",
            "LLM Precision",
            tk.StringVar(value="float16"),
            lambda row: tk.Entry(row, textvariable=self.hp_vars["llm_precision"]).pack(fill=tk.X),
        )
        self._register_hp_row(
            "llm_max_new_tokens",
            "LLM Max New Tokens",
            tk.StringVar(value="120"),
            lambda row: tk.Entry(row, textvariable=self.hp_vars["llm_max_new_tokens"]).pack(fill=tk.X),
        )
        self._register_hp_row(
            "llm_temperature",
            "LLM Temperature",
            tk.StringVar(value="0.4"),
            lambda row: tk.Entry(row, textvariable=self.hp_vars["llm_temperature"]).pack(fill=tk.X),
        )
        self._register_hp_row(
            "llm_top_p",
            "LLM Top P",
            tk.StringVar(value="0.9"),
            lambda row: tk.Entry(row, textvariable=self.hp_vars["llm_top_p"]).pack(fill=tk.X),
        )
        self._register_hp_row(
            "max_history_turns",
            "Max History Turns",
            tk.StringVar(value="6"),
            lambda row: tk.Entry(row, textvariable=self.hp_vars["max_history_turns"]).pack(fill=tk.X),
        )
        self._register_hp_row(
            "audio_sample_rate",
            "Audio Sample Rate",
            tk.StringVar(value="16000"),
            lambda row: tk.Entry(row, textvariable=self.hp_vars["audio_sample_rate"]).pack(fill=tk.X),
        )
        self._register_hp_row(
            "audio_channels",
            "Audio Channels",
            tk.StringVar(value="1"),
            lambda row: tk.Entry(row, textvariable=self.hp_vars["audio_channels"]).pack(fill=tk.X),
        )
        self._register_hp_row(
            "audio_max_seconds",
            "Audio Max Seconds",
            tk.StringVar(value="20"),
            lambda row: tk.Entry(row, textvariable=self.hp_vars["audio_max_seconds"]).pack(fill=tk.X),
        )
        self._register_hp_row(
            "vad_threshold",
            "VAD Threshold",
            tk.StringVar(value="0.5"),
            lambda row: tk.Entry(row, textvariable=self.hp_vars["vad_threshold"]).pack(fill=tk.X),
        )
        self._register_hp_row(
            "vad_min_speech_ms",
            "VAD Min Speech (ms)",
            tk.StringVar(value="250"),
            lambda row: tk.Entry(row, textvariable=self.hp_vars["vad_min_speech_ms"]).pack(fill=tk.X),
        )
        self._register_hp_row(
            "vad_min_silence_ms",
            "VAD Min Silence (ms)",
            tk.StringVar(value="700"),
            lambda row: tk.Entry(row, textvariable=self.hp_vars["vad_min_silence_ms"]).pack(fill=tk.X),
        )
        self._register_hp_row(
            "vad_pre_speech_ms",
            "VAD Pre Speech (ms)",
            tk.StringVar(value="200"),
            lambda row: tk.Entry(row, textvariable=self.hp_vars["vad_pre_speech_ms"]).pack(fill=tk.X),
        )
        self._register_hp_row(
            "auto_stream_max_utterance_s",
            "Auto Stream Max Utterance (s)",
            tk.StringVar(value="20"),
            lambda row: tk.Entry(row, textvariable=self.hp_vars["auto_stream_max_utterance_s"]).pack(fill=tk.X),
        )
        self._register_hp_row(
            "tts_rate",
            "TTS Rate",
            tk.StringVar(value="180"),
            lambda row: tk.Entry(row, textvariable=self.hp_vars["tts_rate"]).pack(fill=tk.X),
        )
        self._register_hp_row(
            "tts_voice_id",
            "TTS Voice ID",
            tk.StringVar(value=""),
            lambda row: tk.Entry(row, textvariable=self.hp_vars["tts_voice_id"]).pack(fill=tk.X),
        )
        self._register_hp_row(
            "tts_stream_min_words",
            "TTS Stream Min Words",
            tk.StringVar(value="4"),
            lambda row: tk.Entry(row, textvariable=self.hp_vars["tts_stream_min_words"]).pack(fill=tk.X),
        )
        self._register_hp_row(
            "kokoro_lang_code",
            "Kokoro Lang Code",
            tk.StringVar(value="a"),
            lambda row: tk.Entry(row, textvariable=self.hp_vars["kokoro_lang_code"]).pack(fill=tk.X),
        )
        self._register_hp_row(
            "kokoro_voice",
            "Kokoro Voice",
            tk.StringVar(value="af_heart"),
            lambda row: tk.Entry(row, textvariable=self.hp_vars["kokoro_voice"]).pack(fill=tk.X),
        )
        self._register_hp_row(
            "kokoro_speed",
            "Kokoro Speed",
            tk.StringVar(value="1.0"),
            lambda row: tk.Entry(row, textvariable=self.hp_vars["kokoro_speed"]).pack(fill=tk.X),
        )
        self._refresh_hyperparam_visibility()

    def _refresh_hyperparam_visibility(self) -> None:
        auto_stream = bool(self.hp_vars.get("auto_stream", tk.BooleanVar(value=True)).get())
        tts_backend = self.tts_var.get().strip().lower()
        stream_mode = self.stream_mode_var.get().strip().lower()

        visible = {
            "hf_local_files_only",
            "auto_stream",
            "whisper_device",
            "whisper_compute_type",
            "llm_device",
            "llm_precision",
            "llm_max_new_tokens",
            "llm_temperature",
            "llm_top_p",
            "max_history_turns",
            "audio_sample_rate",
            "audio_channels",
            "audio_max_seconds",
        }

        if auto_stream:
            visible |= {
                "vad_threshold",
                "vad_min_speech_ms",
                "vad_min_silence_ms",
                "vad_pre_speech_ms",
                "auto_stream_max_utterance_s",
            }

        if tts_backend == "pyttsx3":
            visible |= {"tts_rate", "tts_voice_id"}
        else:
            visible |= {"kokoro_lang_code", "kokoro_voice", "kokoro_speed"}

        if stream_mode == "chunk":
            visible |= {"tts_stream_min_words"}

        for key, row in self.hp_rows.items():
            if key in visible:
                if not row.winfo_ismapped():
                    row.pack(fill=tk.X, pady=3)
            else:
                if row.winfo_ismapped():
                    row.pack_forget()

    def _populate_hyperparam_vars(self, settings: Settings) -> None:
        self.hp_vars["hf_local_files_only"].set(settings.hf_local_files_only)
        self.hp_vars["auto_stream"].set(settings.auto_stream)
        self.hp_vars["whisper_device"].set(settings.whisper_device)
        self.hp_vars["whisper_compute_type"].set(settings.whisper_compute_type)
        self.hp_vars["llm_device"].set(settings.llm_device)
        self.hp_vars["llm_precision"].set(settings.llm_precision)
        self.hp_vars["llm_max_new_tokens"].set(str(settings.llm_max_new_tokens))
        self.hp_vars["llm_temperature"].set(str(settings.llm_temperature))
        self.hp_vars["llm_top_p"].set(str(settings.llm_top_p))
        self.hp_vars["max_history_turns"].set(str(settings.max_history_turns))
        self.hp_vars["audio_sample_rate"].set(str(settings.audio_sample_rate))
        self.hp_vars["audio_channels"].set(str(settings.audio_channels))
        self.hp_vars["audio_max_seconds"].set(str(settings.audio_max_seconds))
        self.hp_vars["vad_threshold"].set(str(settings.vad_threshold))
        self.hp_vars["vad_min_speech_ms"].set(str(settings.vad_min_speech_ms))
        self.hp_vars["vad_min_silence_ms"].set(str(settings.vad_min_silence_ms))
        self.hp_vars["vad_pre_speech_ms"].set(str(settings.vad_pre_speech_ms))
        self.hp_vars["auto_stream_max_utterance_s"].set(str(settings.auto_stream_max_utterance_s))
        self.hp_vars["tts_rate"].set(str(settings.tts_rate))
        self.hp_vars["tts_voice_id"].set(settings.tts_voice_id)
        self.hp_vars["tts_stream_min_words"].set(str(settings.tts_stream_min_words))
        self.hp_vars["kokoro_lang_code"].set(settings.kokoro_lang_code)
        self.hp_vars["kokoro_voice"].set(settings.kokoro_voice)
        self.hp_vars["kokoro_speed"].set(str(settings.kokoro_speed))
        self._refresh_hyperparam_visibility()

    def _runtime_settings(self, base: Settings) -> Settings:
        stream_mode = self.stream_mode_var.get().strip().lower() or base.tts_stream_mode
        tts_streaming = stream_mode != "none"
        effective_mode = "sentence" if stream_mode == "none" else stream_mode
        return replace(
            base,
            vad_provider=self.vad_var.get().strip().lower() or base.vad_provider,
            whisper_model=self.asr_var.get().strip() or base.whisper_model,
            llm_model_id=self.llm_var.get().strip() or base.llm_model_id,
            tts_backend=self.tts_var.get().strip().lower() or base.tts_backend,
            tts_streaming=tts_streaming,
            tts_stream_mode=effective_mode,
            hf_local_files_only=bool(self.hp_vars["hf_local_files_only"].get()),
            auto_stream=bool(self.hp_vars["auto_stream"].get()),
            whisper_device=str(self.hp_vars["whisper_device"].get()).strip() or base.whisper_device,
            whisper_compute_type=str(self.hp_vars["whisper_compute_type"].get()).strip() or base.whisper_compute_type,
            llm_device=str(self.hp_vars["llm_device"].get()).strip() or base.llm_device,
            llm_precision=str(self.hp_vars["llm_precision"].get()).strip() or base.llm_precision,
            llm_max_new_tokens=self._int_from_var("llm_max_new_tokens", base.llm_max_new_tokens),
            llm_temperature=self._float_from_var("llm_temperature", base.llm_temperature),
            llm_top_p=self._float_from_var("llm_top_p", base.llm_top_p),
            max_history_turns=self._int_from_var("max_history_turns", base.max_history_turns),
            audio_sample_rate=self._int_from_var("audio_sample_rate", base.audio_sample_rate),
            audio_channels=self._int_from_var("audio_channels", base.audio_channels),
            audio_max_seconds=self._int_from_var("audio_max_seconds", base.audio_max_seconds),
            vad_threshold=self._float_from_var("vad_threshold", base.vad_threshold),
            vad_min_speech_ms=self._int_from_var("vad_min_speech_ms", base.vad_min_speech_ms),
            vad_min_silence_ms=self._int_from_var("vad_min_silence_ms", base.vad_min_silence_ms),
            vad_pre_speech_ms=self._int_from_var("vad_pre_speech_ms", base.vad_pre_speech_ms),
            auto_stream_max_utterance_s=self._int_from_var(
                "auto_stream_max_utterance_s", base.auto_stream_max_utterance_s
            ),
            tts_rate=self._int_from_var("tts_rate", base.tts_rate),
            tts_voice_id=str(self.hp_vars["tts_voice_id"].get()),
            tts_stream_min_words=self._int_from_var("tts_stream_min_words", base.tts_stream_min_words),
            kokoro_lang_code=str(self.hp_vars["kokoro_lang_code"].get()).strip() or base.kokoro_lang_code,
            kokoro_voice=str(self.hp_vars["kokoro_voice"].get()).strip() or base.kokoro_voice,
            kokoro_speed=self._float_from_var("kokoro_speed", base.kokoro_speed),
        )

    def _load_models(self) -> None:
        if self.is_busy:
            return

        self._set_busy(True)
        self._set_load_progress(0, "Loading selected stack...")

        def worker() -> None:
            old_llm = self.llm
            try:
                base = load_settings()
                self.root.after(0, lambda: self._set_load_progress(1, "Configuration loaded."))

                ready = threading.Event()

                def sync_ui_from_base() -> None:
                    if base.whisper_model in self.ASR_OPTIONS:
                        self.asr_var.set(base.whisper_model)
                    if base.llm_model_id in self.LLM_OPTIONS:
                        self.llm_var.set(base.llm_model_id)
                    if base.vad_provider in self.VAD_OPTIONS:
                        self.vad_var.set(base.vad_provider)
                    if base.tts_backend in self.TTS_OPTIONS:
                        self.tts_var.set(base.tts_backend)
                    mode = "none" if not base.tts_streaming else base.tts_stream_mode
                    if mode in self.STREAM_MODE_OPTIONS:
                        self.stream_mode_var.set(mode)
                    self._populate_hyperparam_vars(base)
                    ready.set()

                self.root.after(0, sync_ui_from_base)
                ready.wait(timeout=2.0)

                runtime = self._runtime_settings(base)
                self.root.after(
                    0,
                    lambda: self._log(
                        "Selected stack: "
                        f"VAD={runtime.vad_provider}, ASR={runtime.whisper_model}, "
                        f"LLM={runtime.llm_model_id}, TTS={runtime.tts_backend}, "
                        f"TTS-stream={runtime.tts_stream_mode}"
                    ),
                )
                self.root.after(0, lambda: self._set_status("Loading Whisper model..."))
                transcriber = WhisperTranscriber(
                    model_name=runtime.whisper_model,
                    device=runtime.whisper_device,
                    compute_type=runtime.whisper_compute_type,
                )
                self.root.after(0, lambda: self._set_load_progress(2, "Whisper model loaded."))

                self.root.after(0, lambda: self._set_status("Loading LLM model..."))
                llm = LocalPhiClient(
                    model_id=runtime.llm_model_id,
                    hf_token=runtime.hf_token,
                    local_files_only=runtime.hf_local_files_only,
                    device=runtime.llm_device,
                    precision=runtime.llm_precision,
                    max_new_tokens=runtime.llm_max_new_tokens,
                    temperature=runtime.llm_temperature,
                    top_p=runtime.llm_top_p,
                )
                self.root.after(0, lambda: self._set_load_progress(3, "LLM model loaded."))

                self.root.after(0, lambda: self._set_status("Initializing TTS..."))
                tts = build_tts(runtime)
                session = ChatSession(
                    system_prompt=runtime.system_prompt,
                    max_history_turns=runtime.max_history_turns,
                )
                self.root.after(0, lambda: self._set_load_progress(4, "TTS initialized."))

                def on_ready() -> None:
                    self._stop_listening(silent=True)
                    self.settings = runtime
                    self.transcriber = transcriber
                    self.llm = llm
                    self.tts = tts
                    self.session = session
                    if old_llm is not None and old_llm is not llm:
                        try:
                            old_llm.close()
                        except Exception:
                            pass
                    self.listen_btn.config(state=tk.NORMAL)
                    self.send_btn.config(state=tk.NORMAL)
                    self._set_busy(False)
                    self._set_status("Models loaded. Listening can start.")
                    self._log("Models loaded successfully.")
                    self._start_listening()

                self.root.after(0, on_ready)
            except Exception as exc:
                self.root.after(
                    0,
                    lambda: (
                        self._set_busy(False),
                        self._set_indicator("error"),
                        self._log(f"Model load failed: {exc}"),
                        self._set_status(f"Failed to load selected stack: {exc}"),
                    ),
                )

        threading.Thread(target=worker, daemon=True).start()

    def _toggle_listening(self) -> None:
        if self.is_busy or self.settings is None or self.transcriber is None:
            return
        if not self.auto_listening:
            self._start_listening()
        else:
            self._stop_listening()

    def _start_listening(self) -> None:
        if self.settings is None or self.transcriber is None or self.auto_listening:
            return
        if self.settings.vad_provider != "silero":
            self._set_indicator("error")
            self._set_status(f"Unsupported VAD provider: {self.settings.vad_provider}")
            self._log("Only Silero VAD is currently implemented.")
            return

        self.auto_listening = True
        self.listen_btn.config(text="Stop Listening")
        self._set_indicator("listening")
        self._set_status("Always-listening started.")
        self._log("Starting always-listening (Silero VAD)...")

        try:
            self.listener = SileroVADListener(
                sample_rate=self.settings.audio_sample_rate,
                threshold=self.settings.vad_threshold,
                min_speech_ms=self.settings.vad_min_speech_ms,
                min_silence_ms=self.settings.vad_min_silence_ms,
                pre_speech_ms=self.settings.vad_pre_speech_ms,
                max_utterance_s=self.settings.auto_stream_max_utterance_s,
            )
        except Exception as exc:
            self.auto_listening = False
            self.listen_btn.config(text="Start Listening")
            self._set_indicator("error")
            self._set_status(f"Could not start listener: {exc}")
            self._log(f"Silero VAD initialization failed: {exc}")
            return

        def listen_worker() -> None:
            assert self.transcriber is not None
            assert self.listener is not None
            try:
                for audio in self.listener.iter_utterances():
                    if not self.auto_listening:
                        break
                    user_text = self.transcriber.transcribe(np.asarray(audio, dtype=np.float32)).strip()
                    if not user_text:
                        continue
                    self._handle_user_text(user_text)
            except Exception as exc:
                self.root.after(0, lambda: self._log(f"Listening error: {exc}"))
                self.root.after(0, lambda: self._set_status(f"Listening stopped: {exc}"))
                self.root.after(0, lambda: self._set_indicator("error"))
            finally:
                self.root.after(0, self._reset_listening_ui)

        threading.Thread(target=listen_worker, daemon=True).start()

    def _stop_listening(self, silent: bool = False) -> None:
        if self.listener is not None:
            self.listener.stop()
        self.auto_listening = False
        self.listen_btn.config(text="Start Listening")
        if not silent:
            self._set_status("Always-listening stopped.")
            self._log("Stopped always-listening.")
        if not self.is_busy:
            self._set_indicator("idle")

    def _reset_listening_ui(self) -> None:
        if self.auto_listening:
            self.auto_listening = False
            self._set_status("Always-listening worker exited.")
            self._log("Always-listening worker exited.")
        self.listen_btn.config(text="Start Listening")
        if not self.is_busy:
            self._set_indicator("idle")

    def _send_text(self) -> None:
        if self.is_busy or self.llm is None:
            return
        text = self.text_input.get().strip()
        if not text:
            return
        self.text_input.delete(0, tk.END)
        threading.Thread(target=lambda: self._handle_user_text(text), daemon=True).start()

    def _handle_user_text(self, user_text: str) -> None:
        if self.llm is None or self.session is None or self.tts is None:
            return
        if self.is_processing_turn:
            return
        self.is_processing_turn = True

        try:
            self.root.after(0, lambda: self._set_indicator("thinking"))
            self.root.after(0, lambda: self._set_status("Generating reply..."))
            self.root.after(0, lambda: self._append_chat(f"\nYou: {user_text}\nAssistant: "))
            self.session.add_user(user_text)

            reply = ""
            try:
                if self.llm_streaming_var.get():
                    reply_chunks: list[str] = []
                    for chunk in self.llm.chat_stream(self.session.messages()):
                        reply_chunks.append(chunk)
                        self.root.after(0, lambda c=chunk: self._append_chat(c))
                    reply = "".join(reply_chunks).strip()
                else:
                    reply = self.llm.chat(self.session.messages()).strip()
                    self.root.after(0, lambda r=reply: self._append_chat(r))
            except Exception as exc:
                self.root.after(0, lambda: self._append_chat(f"\n[llm] {exc}\n"))
                self.root.after(0, lambda: self._set_status("LLM request failed."))
                self.root.after(0, lambda: self._set_indicator("error"))
                return

            if "</think>" in reply:
                reply = reply.split("</think>", 1)[-1].strip()
            if not reply:
                self.root.after(0, lambda: self._append_chat("\n[llm] Empty response.\n"))
                self.root.after(0, lambda: self._set_status("No reply generated."))
                self.root.after(0, lambda: self._set_indicator("error"))
                return

            self.session.add_assistant(reply)
            self.root.after(0, lambda: self._append_chat("\n\n"))
            self.root.after(0, lambda: self._set_indicator("speaking"))
            self.root.after(0, lambda: self._set_status("Speaking..."))
            try:
                if self.listener is not None:
                    self.listener.set_muted(True)
                self.tts.speak(reply)
            except Exception as exc:
                self.root.after(0, lambda: self._append_chat(f"[tts] {exc}\n"))
            finally:
                if self.listener is not None:
                    self.listener.clear_backlog()
                    self.listener.set_muted(False)
            self.root.after(0, lambda: self._set_status("Ready."))
            self.root.after(0, lambda: self._set_indicator("listening" if self.auto_listening else "idle"))
        except Exception as exc:
            self.root.after(0, lambda: self._append_chat(f"[ui] turn processing failed: {exc}\n"))
            self.root.after(0, lambda: self._set_status("Turn processing failed."))
            self.root.after(0, lambda: self._set_indicator("error"))
        finally:
            self.is_processing_turn = False

    def _on_close(self) -> None:
        try:
            self.auto_listening = False
            if self.listener is not None:
                self.listener.stop()
            if self.llm is not None:
                self.llm.close()
        finally:
            self.root.destroy()


def main() -> int:
    root = tk.Tk()
    VoiceAgentUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

