from __future__ import annotations

from pathlib import Path
import sys


EMPOWER_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(EMPOWER_SRC))

from semantic_placement_grounding import semantic_placement_prompt_objects  # noqa: E402


def test_semantic_placement_prompt_objects_include_plan_reference() -> None:
    assert semantic_placement_prompt_objects(
        planning_text="DROP ketchup bottle left to yellow mustard squeeze bottle",
        grasp_object="ketchup bottle",
    ) == ["ketchup bottle", "yellow mustard squeeze bottle"]
