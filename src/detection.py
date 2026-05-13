
import pickle
import cv2
import numpy as np
# import time
import base64
import math
import os
import re
from agents_langchain import Agents
from semantic_placement_grounding import get_semantic_grasp_object
from semantic_placement_grounding import run_grounded_semantic_placement
from semantic_placement_grounding import semantic_placement_prompt_objects
from semantic_placement_grounding import semantic_placement_task_description

class Detection:

    def __init__(self):
        task_description_order = "move the objects on the table to have the objects ordered by height from the highest to lowest"
        task_description_exit = "exit the room"
        task_description_diff = "throw away the objects in the corresponding recycling bin"
        task_description_shelf = "move the objects in the shelf in order to have for each level of the shelf only the objects made of the same material"
        task_description_shelf2 = "move the objects in the shelf in order to have exactly two objects for level"
        task_description_jacket = "give me the green jacket from the clothing rack"
        task_description_semantic_placement = (
            "place the grasped object where it semantically belongs in the scene, "
            "to the left or right of a similar visible object when open shelf space is visible"
        )

        self.task_dict = {
            "order_by_height": task_description_order,
            "exit": task_description_exit,
            "shelf_number": task_description_shelf2,
            "shelf_material": task_description_shelf,
            "recycle": task_description_diff,
            "jacket": task_description_jacket,
            "semantic_placement": task_description_semantic_placement
        }
    
    def run_experiment(self):
        use_case = self.loader_instance.use_case
        image_path = self.loader_instance.SCAN_DIR+"scan.jpg"
        with open(image_path, "rb") as im_file:
            encoded_image = base64.b64encode(im_file.read()).decode("utf-8")

        task_description = self.task_dict[use_case]
        if use_case == "semantic_placement":
            task_description = semantic_placement_task_description(
                get_semantic_grasp_object(self.loader_instance, required=True)
            )

        agents = Agents(encoded_image, task_description)
        # self.single_agent_info = agents.single_agent() 

        environment_agent_info, description_agent_info, planning_agent_info = agents.multi_agent_vision_planning()

        self.results_multi = {
            "environment_agent_info": environment_agent_info,
            "description_agent_info": description_agent_info,
            "planning_agent_info": planning_agent_info,
        }

        with open(self.loader_instance.DUMP_DIR+"planning.pkl",'wb') as f:
            pickle.dump(self.results_multi, f, protocol=2)
        
        with open(self.loader_instance.DUMP_DIR+"planning.txt",'w') as f:
            f.write(self.results_multi["planning_agent_info"])


    def set_loader(self,loader_instance):
        self.loader_instance = loader_instance
        self.run_experiment()
        self.run_image(image_path=self.loader_instance.SCAN_DIR+"scan.jpg")
        if self.loader_instance.use_case == "semantic_placement":
            self.run_semantic_placement()
        
    def split_word(self,words):
        splitted_word = []
        words = words.lower()
        doc = self.loader_instance.nlp(words)
        for token in doc:
            if token.pos_ == "AUX" or (token.pos_ == "NOUN" and token.dep_ in ["dobj","ROOT","nsubj"]) or token.pos_ == "VERB":
                splitted_word.append(token.text)
        return splitted_word

    def compare_two_words(self,list1, list2):
        word1 = list1.copy()
        word2 = list2.copy()
        min_1 = len(word1)
        min_2 = len(word2)
        sim_word = []
        if min_1 > min_2:
            for index in range(min_1):
                found = False
                sim = 0
                min_2 = len(word2)
                for index_2 in range(min_2):
                    try:
                        sim = self.loader_instance.wv.similarity(word1[index], word2[index_2])
                    except KeyError:
                        sim = 0
                    if sim > 0.708:
                        found = True
                        sim_word.append(sim)
                        word2.pop(index_2)
                        break
                if not found:
                    sim_word.append(sim)
        else:
            for index in range(min_2):
                found = False
                sim = 0
                min_1 = len(word1)
                for index_2 in range(min_1):
                    try:
                        sim = self.loader_instance.wv.similarity(word2[index], word1[index_2])
                    except KeyError:
                        sim = 0
                    if sim > 0.708:
                        found = True
                        sim_word.append(sim)
                        word1.pop(index_2)
                        break
                if not found:
                    sim_word.append(sim)
        if sim_word == []:
            return None
        sim_word =  np.mean(sim_word)
        return sim_word

    def is_in_list(self,word,list):
        for obj in list:
            if self.compare_two_words(word, obj) != None and self.compare_two_words(word, obj) > 0.708:
                return True
        return False

    def list_to_prompt_string(self,list):
        list_objects = ""
        for object in list:
            if object != []:
                list_objects += (" ".join(object))
                list_objects += ","
        return list_objects[:-1]

    def list_to_yoloworld(self, list):
        return self.list_to_prompt_string(list)

    def normalize_object_name(self, name):
        if not name:
            return ""
        normalized = name.strip().lower()
        normalized = re.sub(r"^[0-9]+\)\s*", "", normalized)
        normalized = re.sub(r"[^a-z0-9']+", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    def extract_relation_parts(self, relation):
        match = re.search(r"\((.*?)\)", relation)
        if not match:
            return None

        parts = [part.strip() for part in match.group(1).split(",")]
        if len(parts) != 3:
            return None
        return tuple(parts)

    def is_support_object(self, name):
        normalized = self.normalize_object_name(name)
        return normalized in {
            "shelf",
            "table",
            "room",
            "floor",
            "rack",
            "bin",
            "clothing rack",
            "recycling bin",
        }

    def is_action_label(self, name):
        normalized = self.normalize_object_name(name)
        if not normalized:
            return False
        return normalized.split()[0] in {
            "drop",
            "grab",
            "place",
            "navigate",
            "push",
            "pull",
        }

    def is_structural_label(self, name):
        normalized = self.normalize_object_name(name)
        if "shelf" in normalized:
            return True
        return normalized in {
            "cabinet",
            "table",
            "room",
            "floor",
            "rack",
            "bin",
            "clothing rack",
            "recycling bin",
        }

    def extract_scene_objects(self, object_relations):
        seen = set()
        objects = []
        for relation in object_relations:
            parts = self.extract_relation_parts(relation)
            if not parts:
                continue
            for obj_name in (parts[0], parts[2]):
                if self.is_support_object(obj_name) or self.is_action_label(obj_name):
                    continue
                key = self.normalize_object_name(obj_name)
                if key and key not in seen:
                    seen.add(key)
                    objects.append(obj_name.strip())
        return objects

    def extract_object_descriptions(self, planning_text):
        descriptions = {}
        for line in planning_text.splitlines():
            line = line.strip()
            if not line.upper().startswith("GRAB "):
                continue
            object_desc = line[5:].strip()
            canonical_name = object_desc
            if " of " in object_desc.lower():
                canonical_name = object_desc.split(" of ", 1)[1].strip()
            descriptions[self.normalize_object_name(canonical_name)] = object_desc
        return descriptions

    def get_detection_prompts(self, object_relations, planning_text):
        prompts = []
        prompt_to_canonical = {}
        scene_objects = self.extract_scene_objects(object_relations)
        object_descriptions = self.extract_object_descriptions(planning_text)
        if getattr(self.loader_instance, "use_case", "") == "semantic_placement":
            grasp_object = get_semantic_grasp_object(
                self.loader_instance,
                required=False,
            )
            scene_keys = {
                self.normalize_object_name(object_name)
                for object_name in scene_objects
            }
            for object_name in semantic_placement_prompt_objects(
                planning_text=planning_text,
                grasp_object=grasp_object,
            ):
                object_key = self.normalize_object_name(object_name)
                if object_key and object_key not in scene_keys:
                    scene_keys.add(object_key)
                    scene_objects.append(object_name)

        for object_name in scene_objects:
            canonical_name = object_name.strip()
            canonical_key = self.normalize_object_name(canonical_name)
            aliases = [canonical_name]
            object_desc = object_descriptions.get(canonical_key)

            if object_desc:
                aliases.append(object_desc)
                if " of " in object_desc.lower():
                    container, described_name = object_desc.split(" of ", 1)
                    aliases.append(f"{described_name.strip()} {container.strip()}")

            for alias in aliases:
                alias = alias.strip()
                if not alias:
                    continue
                alias_key = self.normalize_object_name(alias)
                if alias_key in prompt_to_canonical:
                    continue
                prompts.append(alias)
                prompt_to_canonical[alias_key] = canonical_name

        if not prompts:
            prompts = ["table"]
            prompt_to_canonical[self.normalize_object_name("table")] = "table"

        return prompts, prompt_to_canonical

    def get_yoloworld_prompts(self, object_relations, planning_text):
        return self.get_detection_prompts(object_relations, planning_text)

    def get_classes(self,object_relations):
        relation_list = []

        for relation in object_relations:
            parts = self.extract_relation_parts(relation)
            if not parts:
                continue
            relation_object_first, _, relation_object_second = parts
            word_1 = self.split_word(relation_object_first)
            word_2 = self.split_word(relation_object_second)
            if not self.is_in_list(word_1,relation_list):
                relation_list.append(word_1)
            if not self.is_in_list(word_2,relation_list):
                relation_list.append(word_2)

        return self.list_to_prompt_string(relation_list)

    def show_mask(self,mask, random_color = True):
        if hasattr(mask, "detach"):
            mask = mask.detach().cpu().numpy()
        mask = np.squeeze(np.asarray(mask))
        if mask.ndim != 2:
            raise ValueError(f"Expected a 2D mask, got shape {mask.shape}")
        mask = mask > 0
        if random_color:
            color = np.concatenate([np.random.random(3)], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask_image = mask_image * 255
        mask_image = mask_image.astype(np.uint8)
        return mask_image

    def box_iou(self, box_a, box_b):
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b
        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
        area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
        union = area_a + area_b - inter_area
        return 0.0 if union == 0 else inter_area / union

    def mask_iou(self, mask_a, mask_b):
        mask_a = np.squeeze(mask_a).astype(bool)
        mask_b = np.squeeze(mask_b).astype(bool)
        intersection = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        return 0.0 if union == 0 else intersection / union

    def sam3_detections(self, sam3_results, prompt_to_canonical):
        candidates = []
        default_max_per_object = (
            "3"
            if getattr(self.loader_instance, "use_case", "") == "semantic_placement"
            else "1"
        )
        max_per_object = int(
            os.environ.get("EMPOWER_SAM3_MAX_PER_OBJECT", default_max_per_object)
        )
        boxes = sam3_results["boxes"]
        scores = sam3_results["scores"]
        class_ids = sam3_results["class_ids"]
        masks = sam3_results["masks"]

        for bbox, score, class_id, mask in zip(boxes, scores, class_ids, masks):
            prompt_label = self.loader_instance.sam3_model.get_class_name(class_id)
            if not prompt_label:
                continue
            label = prompt_to_canonical.get(
                self.normalize_object_name(prompt_label),
                prompt_label,
            )
            if self.is_structural_label(label) or self.is_action_label(label):
                continue
            candidates.append(
                {
                    "bbox": bbox.astype(np.float32),
                    "score": float(score),
                    "label": label,
                    "mask": mask,
                    "prompt": prompt_label,
                }
            )

        candidates.sort(key=lambda item: item["score"], reverse=True)
        kept = []
        label_counts = {}
        for candidate in candidates:
            label_key = self.normalize_object_name(candidate["label"])
            if label_counts.get(label_key, 0) >= max_per_object:
                continue

            duplicate = False
            for existing in kept:
                if (
                    self.mask_iou(candidate["mask"], existing["mask"]) > 0.80
                    or self.box_iou(candidate["bbox"], existing["bbox"]) > 0.85
                ):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
                label_counts[label_key] = label_counts.get(label_key, 0) + 1
        return kept

    def find_bb_relation(self,relation_object):
        index_ = []
        target = self.normalize_object_name(relation_object)
        for index, detection in self.dict_detections.items():
            label = self.normalize_object_name(detection['label'])
            if label == target or label in target or target in label:
                index_.append(index)
        return index_

    def compare_two_list_of_objects(self,position_in_image_first,position_in_image_second,relation,object_first,object_second):
        if position_in_image_first == {} or position_in_image_second == {}:
            return
    
        if "on" in relation:
            min_distance = 30000
            index_first = None
            index_second = None
            for key_first in position_in_image_first.keys():
                for key_second in position_in_image_second.keys():
                    if key_first == key_second:
                        continue
                    dis_x = abs(position_in_image_first[key_first]['x'] - position_in_image_second[key_second]['x'])
                    dis_y = abs(position_in_image_first[key_first]['y'] - position_in_image_second[key_second]['y'])
                    distance = math.sqrt((dis_x*dis_x + dis_y*dis_y))
                    if min_distance > distance:
                        index_first = key_first
                        index_second = key_second
                        min_distance = distance

            if index_first is None or index_second is None:
                return
            self.data_reordered[index_first]['label'] = object_first
            self.data_reordered[index_second]['label'] = object_second
       
        if "left" in relation:
            min_distance_x = 30000
            min_distance_y = 10000
            index_first = None
            index_second = None
            for key_first in position_in_image_first.keys():
                for key_second in position_in_image_second.keys():
                    if key_first == key_second:
                        continue
                    min_distance_x_bb = abs(position_in_image_first[key_first]['x'] - position_in_image_second[key_second]['x'])//2
                    min_distance_y_bb = abs(position_in_image_first[key_first]['y'] - position_in_image_second[key_second]['y'])//2
                    if min_distance_x_bb < min_distance_x and min_distance_x_bb != 0 and min_distance_y > min_distance_y_bb :
                        index_first = key_first
                        index_second = key_second
                        min_distance_x = min_distance_x_bb
                        min_distance_y = min_distance_y_bb

            if index_first is None or index_second is None:
                return
            self.data_reordered[index_first]['label'] = object_first
            self.data_reordered[index_second]['label'] = object_second
            self.dict_detections.pop(index_first, None)
            self.dict_detections.pop(index_second, None)

        if "right" in relation:
            min_distance_x = 30000
            min_distance_y = 10000
            index_first = None
            index_second = None
            for key_first in position_in_image_first.keys():
                for key_second in position_in_image_second.keys():
                    if key_first == key_second:
                        continue
                    min_distance_x_bb = abs(position_in_image_first[key_first]['x'] - position_in_image_second[key_second]['x'])//2
                    min_distance_y_bb = abs(position_in_image_first[key_first]['y'] - position_in_image_second[key_second]['y'])//2
                    if min_distance_x_bb < min_distance_x and min_distance_x_bb != 0 and min_distance_y > min_distance_y_bb:
                        
                        index_first = key_first
                        index_second = key_second
                        min_distance_x = min_distance_x_bb
                        min_distance_y = min_distance_y_bb
            if index_first is None or index_second is None:
                return
            self.data_reordered[index_first]['label'] = object_first
            self.data_reordered[index_second]['label'] = object_second
            self.dict_detections.pop(index_first, None)
            self.dict_detections.pop(index_second, None)

        

    def obtain_bb_grounded(self,index_first,index_second,relation,object_first,object_second):
            detection_data = self.dict_detections
            position_in_image_first = {}
            position_in_image_second = {}
            for i in range(len(index_first)):
                if index_first[i] not in position_in_image_first.keys():
                    position_in_image_first[index_first[i]] = {'x':None,'y':None}
                position_in_image_first[index_first[i]]['x'] = (detection_data[index_first[i]]['bbox'][0] + detection_data[index_first[i]]['bbox'][2]) //2
                position_in_image_first[index_first[i]]['y'] = (detection_data[index_first[i]]['bbox'][1] + detection_data[index_first[i]]['bbox'][3]) //2
            for i in range(len(index_second)):
                if index_second[i] not in position_in_image_second.keys():
                    position_in_image_second[index_second[i]] = {'x':None,'y':None}
                position_in_image_second[index_second[i]]['x'] = (detection_data[index_second[i]]['bbox'][0] + detection_data[index_second[i]]['bbox'][2]) //2
                position_in_image_second[index_second[i]]['y'] = (detection_data[index_second[i]]['bbox'][1] + detection_data[index_second[i]]['bbox'][3]) //2
            self.compare_two_list_of_objects(position_in_image_first,position_in_image_second,relation,object_first,object_second)


    def run_image(self,image_path):
        index_detection = 0
        print("Running detection")
        print(f"Environment agent info: {self.results_multi['environment_agent_info']}")
        object_relations = [
            relation.strip()
            for relation in self.results_multi['environment_agent_info'].split('\n')
            if relation.strip()
        ]
        
        # Check if GPT refused to provide relations
        if "unable" in self.results_multi['environment_agent_info'].lower() or len(object_relations) < 1:
            print("WARNING: GPT did not provide valid relations. Using empty relations.")
            object_relations = []

        # print("Time")
        # start = time.time()
        prompt_labels, prompt_to_canonical = self.get_detection_prompts(
            object_relations,
            self.results_multi.get("planning_agent_info", ""),
        )
        print(f"[DEBUG] SAM3 labels: {prompt_labels}")

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        masked_image = image.copy()
        image_with_bbox = image.copy()
        image_sam3 = image.copy()
        self.dict_detections = {}
        overlay_ = masked_image 
        for filename in os.listdir(self.loader_instance.DUMP_DIR):
            if re.match(r"rgb_[0-9]+\.jpg$", filename):
                os.remove(os.path.join(self.loader_instance.DUMP_DIR, filename))

        score_thr = float(os.environ.get("EMPOWER_SAM3_SCORE_THR", "0.3"))
        self.loader_instance.sam3_model.set_class_name(prompt_labels)
        sam3_results = self.loader_instance.sam3_model.detect(
            image_path,
            max_num_boxes=100,
            score_thr=score_thr,
            nms_thr=0.5,
        )
        detections = self.sam3_detections(sam3_results, prompt_to_canonical)
        print(
            f"[DEBUG] SAM3 raw detections: {len(sam3_results['scores'])} masks, "
            f"kept after canonical dedupe: {len(detections)}"
        )

        for detection in detections:
            bbox = detection["bbox"]
            score = detection["score"]
            label = detection["label"]
            mask = detection["mask"]
            x1, y1, x2, y2 = bbox

            if score > score_thr:
                if index_detection not in self.dict_detections.keys():
                    self.dict_detections[index_detection] = {'bbox':None,'label':None}
                    self.dict_detections[index_detection]['bbox'] = bbox
                    self.dict_detections[index_detection]['label'] = label
                    cv2.rectangle(
                        image_sam3,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (255, 0, 0),
                        1,
                    )
                    cv2.putText(
                        image_sam3,
                        f"{label}: {score:.2f}",
                        (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        1,
                    )
                    
                binary_mask = self.show_mask(mask)
                overlay = masked_image
                self.dict_detections[index_detection]['mask'] = binary_mask
                overlay = cv2.addWeighted(overlay, 1, binary_mask, 0.5, 0)
                cv2.imwrite(self.loader_instance.DUMP_DIR+f"rgb_{index_detection}.jpg", overlay)
                overlay_ = cv2.addWeighted(overlay_, 1, binary_mask, 0.5, 0)
                index_detection += 1
        
        # print("mask : " + str(i) + " : " + str(time.time() -start))
        cv2.imwrite(self.loader_instance.DUMP_DIR+"mask.jpg", overlay_)
        cv2.imwrite(self.loader_instance.DUMP_DIR+"sam3.jpg", image_sam3)
        # cv2.imwrite(self.loader_instance.DUMP_DIR+"yolo.jpg", image_sam3)

        self.data_reordered = self.dict_detections.copy()
        # start = time.time()
        skipped_relations = 0
        for relation in object_relations:
            parts = self.extract_relation_parts(relation)
            if not parts:
                skipped_relations += 1
                continue

            relation_object_first, relation_type, relation_object_second = parts
            if (
                self.is_action_label(relation_object_first)
                or self.is_action_label(relation_object_second)
            ):
                skipped_relations += 1
                continue

            index_bounding_box_first = self.find_bb_relation(relation_object_first)
            index_bounding_box_second = self.find_bb_relation(relation_object_second)
            if index_bounding_box_first != [] or index_bounding_box_second != []:
                self.obtain_bb_grounded(index_bounding_box_first,index_bounding_box_second,relation_type,relation_object_first,relation_object_second)
        if skipped_relations:
            print(f"[DEBUG] Skipped {skipped_relations} malformed relation lines")

        # print("grounfing : " + str(time.time() -start))
        for i, value in self.data_reordered.items():
            cv2.rectangle(image_with_bbox, (int(value['bbox'][0]), int(value['bbox'][1])), (int(value['bbox'][2]), int(value['bbox'][3])), (255, 0, 0), 1 )
            cv2.putText(image_with_bbox, value['label'], (int(value['bbox'][0]+10), int(value['bbox'][1]+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        with open(self.loader_instance.DUMP_DIR+"detection.pkl", 'wb') as f:
            pickle.dump(self.data_reordered, f, protocol=2)
        
        cv2.imwrite(self.loader_instance.DUMP_DIR+"scan_with_bb.jpg", image_with_bbox)

    def run_semantic_placement(self):
        self.semantic_placement_result = run_grounded_semantic_placement(
            loader_instance=self.loader_instance,
            results_multi=self.results_multi,
            detections=self.data_reordered,
        )
