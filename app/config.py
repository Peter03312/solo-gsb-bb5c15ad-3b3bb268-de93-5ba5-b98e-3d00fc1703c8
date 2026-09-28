"""Runtime configuration, sourced from environment variables."""

from __future__ import annotations

import os


class Settings:
    """Simple env-backed settings (no extra dependency)."""

    def __init__(self) -> None:
        # Path of the SQLite database file. Created/initialised at startup.
        self.db_path: str = os.getenv("MSPC_DB_PATH", "./mspc.db")
        # Peaks closer together than this (Da) are merged into one.
        self.merge_tolerance: float = float(os.getenv("MSPC_MERGE_TOLERANCE", "0.0001"))
        # During convolution, peaks below this fraction of the current
        # maximum are pruned to keep the polynomial tractable.
        self.prune_threshold: float = float(os.getenv("MSPC_PRUNE_THRESHOLD", "1e-12"))
        # Upper bound for the top_n request parameter.
        self.top_n_max: int = int(os.getenv("MSPC_TOP_N_MAX", "200"))
        # Default number of peaks returned when top_n is not given.
        self.top_n_default: int = int(os.getenv("MSPC_TOP_N_DEFAULT", "20"))


settings = Settings()
