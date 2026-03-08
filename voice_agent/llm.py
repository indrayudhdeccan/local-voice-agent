from __future__ import annotations

import os
from threading import Thread
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


class LocalPhiClient:
    def __init__(
        self,
        model_id: str,
        hf_token: str = "",
        local_files_only: bool = False,
        device: str = "auto",
        precision: str = "float16",
        max_new_tokens: int = 120,
        temperature: float = 0.4,
        top_p: float = 0.9,
    ) -> None:
        self.model_id = model_id
        self.hf_token = hf_token.strip() or None
        self.local_files_only = local_files_only
        self.device = self._resolve_device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.dtype = self._resolve_dtype(precision)

        if self.local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            token=self.hf_token,
            local_files_only=self.local_files_only,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            token=self.hf_token,
            attn_implementation="eager",
            local_files_only=self.local_files_only,
        )
        self.model.to(self.device)
        self.model.eval()

    def _resolve_device(self, requested: str) -> str:
        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_dtype(self, precision: str) -> torch.dtype:
        if precision == "float16":
            return torch.float16
        if precision == "bfloat16":
            return torch.bfloat16
        if precision == "float32":
            return torch.float32
        raise ValueError(
            "LLM_PRECISION must be one of: float16, bfloat16, float32."
        )

    def _prepare_generation(self, messages: list[dict[str, str]]) -> tuple[dict[str, torch.Tensor], dict]:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        do_sample = self.temperature > 0.0
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p
        return model_inputs, generation_kwargs

    def chat_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        try:
            model_inputs, generation_kwargs = self._prepare_generation(messages)
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            errors: list[Exception] = []

            def generate_worker() -> None:
                try:
                    with torch.inference_mode():
                        self.model.generate(
                            **model_inputs,
                            **generation_kwargs,
                            streamer=streamer,
                        )
                except Exception as exc:
                    errors.append(exc)

            thread = Thread(target=generate_worker, daemon=True)
            thread.start()

            for chunk in streamer:
                if chunk:
                    yield chunk

            thread.join()
            if errors:
                raise errors[0]
        except Exception as exc:
            raise RuntimeError(f"Local LLM generation failed: {exc}") from exc

    def chat(self, messages: list[dict[str, str]]) -> str:
        text = "".join(self.chat_stream(messages)).strip()
        if "</think>" in text:
            text = text.split("</think>", 1)[-1].strip()
        return text

    def close(self) -> None:
        return None

