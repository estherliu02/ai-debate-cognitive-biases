from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def dump_json(path: str | Path, payload: Any):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
