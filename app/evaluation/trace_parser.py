
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_TURN_RE = re.compile(
    r"### Turn \d+\s*\n\s*\*\*User\*\*\s*\n\s*>\s*(?P<user>.+?)\n\s*\*\*Agent\*\*\s*\n(?P<agent>.*?)"
    r"_`end_of_conversation`:\s*\*\*(?P<eoc>true|false)\*\*_",
    re.DOTALL,
)
_URL_RE = re.compile(r"<(https://www\.shl\.com/[^>]+)>")


@dataclass
class TraceTurn:
    user_message: str
    agent_reply: str
    recommended_urls: list[str]
    end_of_conversation: bool


@dataclass
class ConversationTrace:
    trace_id: str
    turns: list[TraceTurn]

    @property
    def expected_shortlist_urls(self) -> set[str]:
        """URLs from the last turn that produced a non-empty shortlist."""
        for turn in reversed(self.turns):
            if turn.recommended_urls:
                return set(turn.recommended_urls)
        return set()


def parse_trace_file(path: str | Path) -> ConversationTrace:
    text = Path(path).read_text(encoding="utf-8")
    turns: list[TraceTurn] = []

    for match in _TURN_RE.finditer(text):
        agent_block = match.group("agent")
        urls = _URL_RE.findall(agent_block)
        turns.append(
            TraceTurn(
                user_message=match.group("user").strip(),
                agent_reply=agent_block.strip(),
                recommended_urls=urls,
                end_of_conversation=(match.group("eoc") == "true"),
            )
        )

    return ConversationTrace(trace_id=Path(path).stem, turns=turns)


def load_all_traces(directory: str | Path) -> list[ConversationTrace]:
    directory = Path(directory)
    return [parse_trace_file(p) for p in sorted(directory.glob("*.md"))]
