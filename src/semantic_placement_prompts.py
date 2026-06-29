"""LLM prompt helpers for semantic placement modes."""

from __future__ import annotations

SEMANTIC_PLACEMENT_ENVIRONMENT_TASK = (
    "place the grasped object where it semantically belongs in the scene"
)

SEMANTIC_PLACEMENT_PLANNING_TASK = (
    "place the grasped object where it semantically belongs in the scene, "
    "using only one DROP action with a relation such as left, right, "
    "or on and a reference object, such as "
    "'DROP grasped object left to cereal box'"
)


def _personalize_grasp_object(task_description: str, grasp_object: str) -> str:
    return task_description.replace("grasped object", grasp_object)


def semantic_placement_environment_task_description_refined(grasp_object: str) -> str:
    return (
        f"The robot is already holding this object: {grasp_object}. "
        "The held object is not visible in the image. Use the image only as "
        "the placement scene. List spatial relations between visible movable "
        "objects on the placement surface that help decide where the held "
        "object semantically belongs, such as placing a carton next to one "
        "visible carton or a condiment next to one visible condiment. If a "
        "visible object has the same type or name as the held object, include "
        "that object and its neighbors in the relations."
    )


def semantic_placement_planning_task_description_refined(grasp_object: str) -> str:
    return (
        f"The robot is already holding this object: {grasp_object}. "
        "The held object is not visible in the image. Use the image only as "
        "the placement scene. Choose exactly one visible reference object "
        "where the held object semantically belongs. If a visible object has "
        "the same type or name as the held object, use one same-type object "
        "as the reference before choosing any other category. Choose only a "
        "LEFT or RIGHT placement relative to that single visible reference "
        "object, and choose the side that has open shelf space in the image. "
        "Do not use behind, in front of, near, beside, into, on, group, "
        "area, shelf, or category as the placement relation/reference. "
        "Use only one action line: DROP. The reference object in the DROP "
        "line must use exactly one object name from the environment-relation "
        "triples. Return exactly one action line such as "
        f"'DROP {grasp_object} left to cereal box' or 'DROP {grasp_object} "
        "right to cereal box'."
    )


def semantic_placement_environment_task_description_empower(grasp_object: str) -> str:
    """High-level task goal passed to the environment agent, like original main."""
    del grasp_object
    return SEMANTIC_PLACEMENT_ENVIRONMENT_TASK


def semantic_placement_planning_task_description_empower(grasp_object: str) -> str:
    """Planning task from original semantic_placement detection task dict."""
    return _personalize_grasp_object(SEMANTIC_PLACEMENT_PLANNING_TASK, grasp_object)


def semantic_placement_task_descriptions(
    grasp_object: str,
    *,
    refined: bool,
) -> tuple[str, str]:
    if refined:
        return (
            semantic_placement_environment_task_description_refined(grasp_object),
            semantic_placement_planning_task_description_refined(grasp_object),
        )
    return (
        semantic_placement_environment_task_description_empower(grasp_object),
        semantic_placement_planning_task_description_empower(grasp_object),
    )


def semantic_placement_refined_task_description(grasp_object: str) -> str:
    """Planning-task description for refined semantic placement."""
    return semantic_placement_planning_task_description_refined(grasp_object)


def semantic_placement_empower_task_description(grasp_object: str) -> str:
    """Planning-task description for empower semantic placement."""
    return semantic_placement_planning_task_description_empower(grasp_object)


__all__ = [
    "SEMANTIC_PLACEMENT_ENVIRONMENT_TASK",
    "SEMANTIC_PLACEMENT_PLANNING_TASK",
    "semantic_placement_empower_task_description",
    "semantic_placement_environment_task_description_empower",
    "semantic_placement_environment_task_description_refined",
    "semantic_placement_planning_task_description_empower",
    "semantic_placement_planning_task_description_refined",
    "semantic_placement_refined_task_description",
    "semantic_placement_task_descriptions",
]
