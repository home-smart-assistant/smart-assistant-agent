from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass
class MemoryTurn:
    role: str
    content: str


class ShortTermMemory:
    def __init__(self, max_turns: int, token_budget: int) -> None:
        self._max_turns = max(2, max_turns)
        self._token_budget = max(200, token_budget)
        self._sessions: dict[str, deque[MemoryTurn]] = defaultdict(lambda: deque(maxlen=self._max_turns))

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        self._sessions[session_id].append(MemoryTurn(role=role, content=content.strip()))

    def get_session(self, session_id: str) -> list[MemoryTurn]:
        return list(self._sessions.get(session_id, []))

    def get_window(self, session_id: str, max_history_turns: int) -> list[MemoryTurn]:
        turns = list(self._sessions.get(session_id, []))
        if max_history_turns > 0:
            turns = turns[-(max_history_turns * 2) :]

        pruned: list[MemoryTurn] = []
        token_total = 0
        for turn in reversed(turns):
            current = estimate_tokens(turn.content) + 2
            if token_total + current > self._token_budget and pruned:
                break
            token_total += current
            pruned.append(turn)
        pruned.reverse()
        return pruned

    def clear_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def session_count(self) -> int:
        return len(self._sessions)
