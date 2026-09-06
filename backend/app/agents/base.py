"""Agent primitives: tool-call tracing."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Step:
    tool: str
    args: dict[str, Any]
    status: str = "ok"  # ok | failed | skipped
    observation: str = ""
    duration_s: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = {
            "tool": self.tool,
            "args": self.args,
            "status": self.status,
            "observation": self.observation,
            "duration_s": round(self.duration_s, 3),
        }
        if self.error:
            d["error"] = self.error
        return d


@dataclass
class AgentTrace:
    steps: list[Step] = field(default_factory=list)

    def record(self, tool: str, args: dict[str, Any], observation: str = "",
               status: str = "ok", error: str = "", duration: float | None = None) -> Step:
        step = Step(
            tool=tool,
            args=args,
            status=status,
            observation=observation,
            duration_s=duration if duration is not None else 0.0,
            error=error,
        )
        self.steps.append(step)
        return step

    def to_dict(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.steps]


class ToolTimer:
    """Context manager that measures a tool call and records it in a trace."""

    def __init__(self, trace: AgentTrace, tool: str, args: dict[str, Any]):
        self.trace = trace
        self.tool = tool
        self.args = args
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def ok(self, observation: str) -> Step:
        return self.trace.record(self.tool, self.args, observation=observation,
                                 status="ok", duration=time.perf_counter() - self.t0)

    def fail(self, error: str) -> Step:
        return self.trace.record(self.tool, self.args, status="failed", error=error,
                                 duration=time.perf_counter() - self.t0)
