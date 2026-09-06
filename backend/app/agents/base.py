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

    def __init__(self, trace: AgentTrace, tool: str, args: dict[str, Any],
                 dataset_id: str | None = None, label: str | None = None):
        self.trace = trace
        self.tool = tool
        self.args = args
        self.t0 = 0.0
        # When a dataset is given the step is also published live, so the UI can
        # show which stage is running instead of a generic spinner.
        self.dataset_id = dataset_id
        self.label = label or tool

    def __enter__(self):
        self.t0 = time.perf_counter()
        if self.dataset_id:
            from . import progress

            progress.begin_stage(self.dataset_id, self.label,
                                 ", ".join(f"{k}={v}" for k, v in list(self.args.items())[:3]))
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def ok(self, observation: str, status: str = "ok") -> Step:
        """Record a completed step. `status` allows "skipped" for a step that
        legitimately had nothing to do — passing it used to raise TypeError and
        turn a harmless skip into a failed run."""
        seconds = time.perf_counter() - self.t0
        if self.dataset_id:
            from . import progress

            progress.end_stage(self.dataset_id, self.label, observation, seconds)
        return self.trace.record(self.tool, self.args, observation=observation,
                                 status=status, duration=seconds)

    def fail(self, error: str) -> Step:
        return self.trace.record(self.tool, self.args, status="failed", error=error,
                                 duration=time.perf_counter() - self.t0)
