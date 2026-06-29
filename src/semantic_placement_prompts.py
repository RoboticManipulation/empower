"""LLM prompt helpers for semantic placement modes."""

from __future__ import annotations


def semantic_placement_refined_task_description(grasp_object: str) -> str:
    return (
        f"The robot is already holding this object: {grasp_object}. "
        "The held object is not visible in the image. Use the image only as "
        "the placement scene. Choose exactly one visible reference object "
        "where the held object semantically belongs, such as placing a carton "
        "next to one visible carton or a condiment next to one visible "
        "condiment. If a visible object has the same type or name as the held "
        "object, use one same-type object as the reference before choosing any "
        "other category. Choose only a LEFT or RIGHT placement relative to "
        "that single visible reference object, and choose the side that has "
        "open shelf space in the image. Do not use behind, in front of, near, "
        "beside, into, on, group, area, shelf, or category as the placement "
        "relation/reference. The reference object in the DROP line must use "
        "exactly one object name from the environment-relation triples. "
        "Return exactly one action line such as "
        f"'DROP {grasp_object} left to cereal box' or 'DROP {grasp_object} "
        "right to cereal box'."
    )


def semantic_placement_empower_task_description(grasp_object: str) -> str:
    return (
        f"The robot is already holding {grasp_object}. "
        "Use the image as the placement scene and place the held object where "
        "it semantically belongs.\n"
        "Use only one action line: DROP. DROP places the held object "
        "with respect to a visible reference "
        "object, for example "
        f"'DROP {grasp_object} left to cereal box', "
        f"'DROP {grasp_object} right to cereal box', or "
        f"'DROP {grasp_object} on cereal box'.\n"
        "Use only these placement relations: left, right, or on. The reference "
        "object in the DROP line must use exactly one object name from the "
        "environment-relation triples. Return only the DROP action line and "
        "nothing else."
    )


__all__ = [
    "semantic_placement_empower_task_description",
    "semantic_placement_refined_task_description",
]
