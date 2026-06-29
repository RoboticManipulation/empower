from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semantic_placement_prompts import (
    SEMANTIC_PLACEMENT_ENVIRONMENT_TASK,
    SEMANTIC_PLACEMENT_PLANNING_TASK,
    semantic_placement_task_descriptions,
)


class SemanticPlacementPromptTests(unittest.TestCase):
    def test_refined_environment_prompt_excludes_drop_instructions(self) -> None:
        environment_task, planning_task = semantic_placement_task_descriptions(
            "ketchup bottle",
            refined=True,
        )
        self.assertNotIn("Use only one action line: DROP", environment_task)
        self.assertNotIn("Return exactly one action line", environment_task)
        self.assertIn("Use only one action line: DROP", planning_task)

    def test_empower_prompts_match_original_main_task_descriptions(self) -> None:
        environment_task, planning_task = semantic_placement_task_descriptions(
            "ketchup bottle",
            refined=False,
        )
        self.assertEqual(environment_task, SEMANTIC_PLACEMENT_ENVIRONMENT_TASK)
        self.assertEqual(
            planning_task,
            SEMANTIC_PLACEMENT_PLANNING_TASK.replace("grasped object", "ketchup bottle"),
        )
        self.assertNotIn("DROP ketchup bottle", environment_task)

    def test_refined_planning_prompt_requires_environment_relation_names(self) -> None:
        _, planning_task = semantic_placement_task_descriptions(
            "lip balm tube",
            refined=True,
        )
        self.assertIn("environment-relation triples", planning_task)
        self.assertIn("lip balm tube", planning_task)


if __name__ == "__main__":
    unittest.main()
