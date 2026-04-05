"""Server-side session state manager.

Provides a thread-safe store for uploaded datasets, cleaned data,
analytics results, and pipeline artifacts. Designed for multi-user
scalability via session-keyed storage.
"""

import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import pandas as pd

_lock = threading.Lock()
_sessions: dict[str, dict[str, Any]] = {}

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_SESSION_AGE = 3600 * 4  # 4 hours


def create_session() -> str:
    session_id = str(uuid.uuid4())
    with _lock:
        _sessions[session_id] = {
            "created": time.time(),
            "datasets_raw": {},      # {source_name: DataFrame}
            "datasets_cleaned": {},  # {source_name: DataFrame}
            "integrated": None,      # DataFrame
            "featured": None,        # DataFrame
            "analytics": {},         # {key: result_dict}
            "figures": {},           # {name: plotly Figure}
            "pipeline_log": [],      # list of log strings
            "config": None,          # pipeline config dict
        }
    return session_id


def get(session_id: str, key: str, default=None) -> Any:
    with _lock:
        session = _sessions.get(session_id, {})
        return session.get(key, default)


def put(session_id: str, key: str, value: Any):
    with _lock:
        if session_id not in _sessions:
            _sessions[session_id] = {"created": time.time()}
        _sessions[session_id][key] = value


def log(session_id: str, message: str):
    with _lock:
        if session_id in _sessions:
            _sessions[session_id].setdefault("pipeline_log", []).append(
                f"{time.strftime('%H:%M:%S')} {message}"
            )


def get_log(session_id: str) -> list[str]:
    with _lock:
        return list(_sessions.get(session_id, {}).get("pipeline_log", []))


def list_datasets(session_id: str) -> dict[str, int]:
    """Return {source_name: row_count} for raw datasets."""
    with _lock:
        raw = _sessions.get(session_id, {}).get("datasets_raw", {})
        return {k: len(v) for k, v in raw.items()}


def cleanup_old_sessions():
    now = time.time()
    with _lock:
        expired = [
            sid for sid, data in _sessions.items()
            if now - data.get("created", 0) > MAX_SESSION_AGE
        ]
        for sid in expired:
            del _sessions[sid]
