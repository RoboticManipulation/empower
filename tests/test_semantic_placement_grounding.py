from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from detection import Detection


class OpenVocabPromptTests(unittest.TestCase):
    def setUp(self):
        self.detection = Detection()
        self.detection.loader_instance = MagicMock()
        self.detection.split_word = lambda words: words.lower().split()
        self.detection.is_in_list = self._exact_word_list_match

    @staticmethod
    def _exact_word_list_match(word_list, existing_lists):
        for existing in existing_lists:
            if word_list == existing:
                return True
        return False

    def test_build_open_vocab_prompts_from_environment_relations(self):
        object_relations = [
            "(toothpaste tube, on, shelf)",
            "(soup can, left to, juice box)",
        ]
        prompts, prompt_to_canonical = self.detection.build_open_vocab_prompts(
            object_relations
        )
        self.assertEqual(
            prompts,
            ["toothpaste tube", "soup can", "juice box"],
        )
        self.assertEqual(prompt_to_canonical["toothpaste tube"], "toothpaste tube")

    def test_build_open_vocab_prompts_ignore_action_lines(self):
        object_relations = [
            "DROP ketchup_bottle left to mustard",
            "(shampoo bottle, on, shelf)",
        ]
        prompts, _ = self.detection.build_open_vocab_prompts(object_relations)
        self.assertEqual(prompts, ["shampoo bottle"])

    def test_build_open_vocab_prompts_exclude_grasp_object(self):
        object_relations = [
            "(ketchup bottle, on, shelf)",
            "(soup can, left to, juice box)",
        ]
        prompts, _ = self.detection.build_open_vocab_prompts(
            object_relations,
            exclude_objects=["ketchup bottle"],
        )
        self.assertEqual(prompts, ["soup can", "juice box"])

    def test_get_classes_matches_comma_joined_prompts(self):
        object_relations = [
            "(toothpaste tube, on, shelf)",
            "(soup can, left to, juice box)",
        ]
        self.assertEqual(
            self.detection.get_classes(object_relations),
            "toothpaste tube,soup can,juice box",
        )

    def test_find_bb_relation_matches_detection_label_in_relation_object(self):
        self.detection.dict_detections = {
            0: {"label": "jar", "bbox": [0, 0, 1, 1]},
            1: {"label": "box", "bbox": [2, 2, 3, 3]},
        }
        matched = self.detection.find_bb_relation("Jif peanut butter jar")
        self.assertEqual(matched, [0])
