"""Empower: semantic placement and manipulation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

__version__ = "0.0.1"

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from empower.semantic_placement_wrapper import EmpowerSemanticPlacementWrapper  # noqa: E402

__all__ = ["EmpowerSemanticPlacementWrapper", "__version__"]
