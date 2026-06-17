from __future__ import annotations

from empower.semantic_placement_prompts import semantic_placement_prompt_objects


def test_semantic_placement_prompt_objects_include_plan_reference() -> None:
    assert semantic_placement_prompt_objects(
        planning_text="DROP ketchup bottle left to yellow mustard squeeze bottle",
        grasp_object="ketchup bottle",
    ) == ["ketchup bottle", "yellow mustard squeeze bottle"]
