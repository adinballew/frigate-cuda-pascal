#!/usr/bin/env python3
"""Compatibility wrapper for the canonical Frigate+ shadow importer.

The implementation lives in:

    /opt/frigate/custom-model/frigate_plus_shadow.py

Keep this wrapper so old co-located labeler commands fail less painfully and
there is only one real implementation to maintain.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

CANONICAL = Path(__file__).resolve().parents[1] / "frigate-custom-model" / "frigate_plus_shadow.py"

if __name__ == "__main__":
    if not CANONICAL.is_file():
        raise SystemExit(f"canonical Frigate+ shadow script not found: {CANONICAL}")
    sys.argv[0] = str(CANONICAL)
    runpy.run_path(str(CANONICAL), run_name="__main__")
