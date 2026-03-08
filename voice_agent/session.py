from __future__ import annotations

from collections import deque
from typing import Deque


class ChatSession:
    def __init__(self, system_prompt: str, max_history_turns: int = 6) -> None:
        self.system_prompt = system_prompt
        self._messages: Deque[dict[str, str]] = deque(maxlen=max_history_turns * 2)

    def add_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})

    def messages(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system_prompt}, *list(self._messages)]

