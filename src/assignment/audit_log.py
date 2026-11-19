"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}
        self._inputs: dict[str, str] = {}

    @staticmethod
    def _key(user_id: str, request_id: str | None) -> str:
        return request_id or user_id

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Remember the input text and start time until the matching output arrives."""
        key = self._key(user_id, request_id)
        self._open[key] = time.perf_counter()
        self._inputs[key] = text

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Close the request: store output, decision, layer and latency."""
        key = self._key(user_id, request_id)
        started = self._open.pop(key, None)
        latency_ms = (
            round((time.perf_counter() - started) * 1000, 2)
            if started is not None
            else None
        )
        entry = {
            "timestamp": utc_now_iso(),
            "request_id": request_id,
            "user_id": user_id,
            "input": self._inputs.pop(key, None),
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "latency_ms": latency_ms,
        }
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
