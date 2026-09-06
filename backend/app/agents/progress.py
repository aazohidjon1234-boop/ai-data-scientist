"""Live progress for a running pipeline.

Training is a single long POST, so until it returns the UI can only show a
generic "working…". The agent already times every tool call for its trace; this
publishes that same information *while* it happens, keyed by dataset, so the
browser can poll and show the current stage and what is already finished.

Deliberately in-memory: it is a progress indicator, not a record. Losing it on
restart costs nothing, and the authoritative trace is still stored with the run.
"""
from __future__ import annotations

import threading
import time
from typing import Any

# dataset_id -> live state. Small and bounded by the number of concurrent runs.
_RUNS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()
MAX_TRACKED = 64


def start(dataset_id: str, phase: str, stages: list[str]) -> None:
    with _LOCK:
        if len(_RUNS) >= MAX_TRACKED:
            oldest = min(_RUNS, key=lambda k: _RUNS[k]["started"])
            _RUNS.pop(oldest, None)
        _RUNS[dataset_id] = {
            "phase": phase,
            "planned": stages,
            "current": stages[0] if stages else None,
            "done": [],
            "started": time.time(),
            "stage_started": time.time(),
            "finished": False,
            "error": None,
        }


def begin_stage(dataset_id: str, stage: str, detail: str = "") -> None:
    with _LOCK:
        run = _RUNS.get(dataset_id)
        if not run:
            return
        run["current"] = stage
        run["detail"] = detail
        run["stage_started"] = time.time()


def end_stage(dataset_id: str, stage: str, observation: str, seconds: float) -> None:
    with _LOCK:
        run = _RUNS.get(dataset_id)
        if not run:
            return
        run["done"].append({
            "stage": stage,
            "observation": observation[:300],
            "seconds": round(seconds, 2),
        })


def finish(dataset_id: str, error: str | None = None) -> None:
    with _LOCK:
        run = _RUNS.get(dataset_id)
        if not run:
            return
        run["finished"] = True
        run["current"] = None
        run["error"] = error


def snapshot(dataset_id: str) -> dict[str, Any]:
    with _LOCK:
        run = _RUNS.get(dataset_id)
        if not run:
            return {"running": False, "phase": None, "current": None, "done": [], "planned": []}
        now = time.time()
        planned = list(run["planned"])
        # Count position in the PLAN, not raw tool calls: training fires one
        # call per model, which would otherwise report things like "13 of 8".
        if run["finished"]:
            completed = len(planned)
        elif run["current"] in planned:
            completed = planned.index(run["current"])
        else:
            reached = [planned.index(d["stage"]) for d in run["done"] if d["stage"] in planned]
            completed = (max(reached) + 1) if reached else 0
        return {
            "running": not run["finished"],
            "phase": run["phase"],
            "current": run["current"],
            "detail": run.get("detail", ""),
            "planned": planned,
            "done": list(run["done"]),
            "completed": completed,
            "total": len(planned),
            "elapsed": round(now - run["started"], 1),
            "stage_elapsed": round(now - run["stage_started"], 1),
            "finished": run["finished"],
            "error": run["error"],
        }
