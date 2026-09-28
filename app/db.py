"""SQLite persistence: schema initialisation and an envelope-result cache.

The database file (and schema) is created automatically at application
startup; no manual migration step is required.
"""

from __future__ import annotations

import json
import sqlite3
import threading

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS envelope_cache (
    formula    TEXT    NOT NULL,
    top_n      INTEGER NOT NULL,
    result     TEXT    NOT NULL,
    created_at TEXT    NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (formula, top_n)
);
CREATE TABLE IF NOT EXISTS request_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    formula    TEXT    NOT NULL,
    top_n      INTEGER,
    ok         INTEGER NOT NULL,
    message    TEXT,
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""


def init_db(db_path: str) -> None:
    """Open the database (creating the file if needed) and apply the schema."""
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
        _conn = sqlite3.connect(db_path, check_same_thread=False)
        _conn.executescript(_SCHEMA)
        _conn.commit()


def close_db() -> None:
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None


def _require_conn() -> sqlite3.Connection:
    if _conn is None:
        raise RuntimeError("Database not initialised; init_db() must run at startup")
    return _conn


def get_cached(formula: str, top_n: int) -> dict | None:
    with _lock:
        row = _require_conn().execute(
            "SELECT result FROM envelope_cache WHERE formula = ? AND top_n = ?",
            (formula, top_n),
        ).fetchone()
    return json.loads(row[0]) if row else None


def put_cache(formula: str, top_n: int, result: dict) -> None:
    with _lock:
        conn = _require_conn()
        conn.execute(
            "INSERT OR REPLACE INTO envelope_cache (formula, top_n, result) "
            "VALUES (?, ?, ?)",
            (formula, top_n, json.dumps(result)),
        )
        conn.commit()


def log_request(formula: str, top_n: int | None, ok: bool, message: str | None) -> None:
    with _lock:
        conn = _require_conn()
        conn.execute(
            "INSERT INTO request_log (formula, top_n, ok, message) VALUES (?, ?, ?, ?)",
            (formula, top_n, int(ok), message),
        )
        conn.commit()
