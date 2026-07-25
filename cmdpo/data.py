from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class PreferenceExample:
    prompt: str
    chosen: str
    rejected: str
    answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "PreferenceExample":
        return cls(
            prompt=item["prompt"],
            chosen=item["chosen"],
            rejected=item["rejected"],
            answer=item.get("answer"),
            metadata=item.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "answer": self.answer,
            "metadata": self.metadata,
        }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
