from __future__ import annotations

from datetime import datetime
import uuid


def make_run_id() -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    short = uuid.uuid4().hex[:8]
    return f"run_{ts}_{short}"
