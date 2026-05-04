import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import numpy as np
import cv2
import torch.nn.functional as F

class VitSam():

    def __init__(self, encoder_model, decoder_model):
        from efficientvit.export_encoder import SamResize
        from efficientvit.inference import SamDecoder, SamEncoder
        import torchvision.transforms as transforms

        self.decoder = SamDecoder(decoder_model)
        self.encoder = SamEncoder(encoder_model)
        self.resize_transform_cls = SamResize
        self.transforms = transforms


    def __call__(self, img, bboxes):
        raw_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        origin_image_size = raw_img.shape[:2]
        img = self._preprocess(raw_img, img_size=512)
        img_embeddings = self.encoder(img)
        boxes = np.array(bboxes, dtype=np.float32)
        masks, _, _ = self.decoder.run(
            img_embeddings=img_embeddings,
            origin_image_size=origin_image_size,
            boxes=boxes,
        )

        return masks, boxes

    def _preprocess(self, x, img_size=512):
        pixel_mean = [123.675 / 255, 116.28 / 255, 103.53 / 255]
        pixel_std = [58.395 / 255, 57.12 / 255, 57.375 / 255]

        x = torch.tensor(x)
        resize_transform = self.resize_transform_cls(img_size)
        x = resize_transform(x).float() / 255
        x = self.transforms.Normalize(mean=pixel_mean, std=pixel_std)(x)

        h, w = x.shape[-2:]
        th, tw = img_size, img_size
        assert th >= h and tw >= w
        x = F.pad(x, (0, tw - w, 0, th - h), value=0).unsqueeze(0).numpy()

        return x
    


class SAM3Detector:
    """Prompt-conditioned detector/segmenter backed by SAM3."""

    def __init__(
        self,
        checkpoint_path=None,
        device=None,
        confidence_threshold=0.5,
        use_bfloat16=True,
    ):
        self.objects = []
        self.class_names = ""
        self.device = device or os.environ.get(
            "EMPOWER_SAM3_DEVICE",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        self.confidence_threshold = float(
            os.environ.get("EMPOWER_SAM3_SCORE_THR", confidence_threshold)
        )
        self.use_bfloat16 = use_bfloat16

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        build_sam3_image_model, Sam3Processor = self._import_sam3()
        checkpoint_path = self._resolve_checkpoint(checkpoint_path)
        self.model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
            device=self.device,
            eval_mode=True,
        )
        self.processor = Sam3Processor(
            self.model,
            device=self.device,
            confidence_threshold=self.confidence_threshold,
        )

    def _import_sam3(self):
        try:
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ModuleNotFoundError:
            sam3_root = Path(__file__).resolve().parents[2] / "sam3"
            if not sam3_root.exists():
                raise
            sys.path.insert(0, str(sam3_root))
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        return build_sam3_image_model, Sam3Processor

    def _resolve_checkpoint(self, checkpoint_path):
        checkpoint_path = checkpoint_path or os.environ.get("EMPOWER_SAM3_CHECKPOINT")
        if checkpoint_path:
            return checkpoint_path

        from huggingface_hub import hf_hub_download

        token_present = bool(
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_HUB_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        try:
            return hf_hub_download(
                repo_id="facebook/sam3",
                filename="sam3.pt",
                local_files_only=not token_present,
            )
        except Exception as exc:
            if token_present:
                raise
            raise RuntimeError(
                "SAM3 checkpoint is not cached and no Hugging Face token is visible "
                "to this Python process. Export HF_TOKEN, run `hf auth login`, or set "
                "EMPOWER_SAM3_CHECKPOINT=/path/to/sam3.pt."
            ) from exc

    def _autocast(self):
        if self.device == "cuda" and self.use_bfloat16:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _load_image(self, input_image):
        from PIL import Image

        if isinstance(input_image, (str, os.PathLike)):
            return Image.open(input_image).convert("RGB")
        if isinstance(input_image, np.ndarray):
            if input_image.ndim != 3:
                raise ValueError("SAM3 expects a color image")
            return Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
        return input_image.convert("RGB")

    def set_class_name(self, objects):
        if isinstance(objects, (list, tuple)):
            self.objects = [obj.strip() for obj in objects if obj and obj.strip()]
            self.class_names = ",".join(self.objects)
        else:
            self.class_names = objects
            self.objects = [obj.strip() for obj in objects.split(",") if obj.strip()]

    def get_class_name(self, id):
        id = int(id)
        if id < 0 or id >= len(self.objects):
            return None
        return self.objects[id]

    def _classwise_nms(self, boxes, scores, class_ids, masks, nms_thr):
        if len(scores) == 0:
            return boxes, scores, class_ids, masks
        try:
            from torchvision.ops import nms
        except Exception:
            return boxes, scores, class_ids, masks

        keep_indices = []
        for class_id in sorted(set(class_ids.tolist())):
            indices = np.flatnonzero(class_ids == class_id)
            keep = nms(
                torch.as_tensor(boxes[indices], dtype=torch.float32),
                torch.as_tensor(scores[indices], dtype=torch.float32),
                float(nms_thr),
            )
            keep_indices.extend(indices[keep.cpu().numpy()].tolist())

        keep_indices = sorted(
            keep_indices, key=lambda idx: float(scores[idx]), reverse=True
        )
        return (
            boxes[keep_indices],
            scores[keep_indices],
            class_ids[keep_indices],
            masks[keep_indices],
        )

    def detect(self, input_image, max_num_boxes=100, score_thr=None, nms_thr=0.5):
        if not self.objects:
            return {
                "boxes": np.empty((0, 4), dtype=np.float32),
                "scores": np.empty((0,), dtype=np.float32),
                "class_ids": np.empty((0,), dtype=np.int64),
                "masks": np.empty((0, 0, 0), dtype=bool),
            }

        image = self._load_image(input_image)
        threshold = self.confidence_threshold if score_thr is None else float(score_thr)
        self.processor.confidence_threshold = threshold

        boxes, scores, class_ids, masks = [], [], [], []
        with self._autocast():
            state = self.processor.set_image(image)
            for class_id, prompt in enumerate(self.objects):
                self.processor.reset_all_prompts(state)
                result = self.processor.set_text_prompt(state=state, prompt=prompt)
                prompt_boxes = result["boxes"].detach().float().cpu().numpy()
                prompt_scores = result["scores"].detach().float().cpu().numpy()
                prompt_masks = result["masks"].detach().cpu().numpy()

                for box, score, mask in zip(prompt_boxes, prompt_scores, prompt_masks):
                    if float(score) < threshold:
                        continue
                    boxes.append(box.astype(np.float32))
                    scores.append(np.float32(score))
                    class_ids.append(np.int64(class_id))
                    masks.append(np.squeeze(mask).astype(bool))

        if not scores:
            width, height = image.size
            return {
                "boxes": np.empty((0, 4), dtype=np.float32),
                "scores": np.empty((0,), dtype=np.float32),
                "class_ids": np.empty((0,), dtype=np.int64),
                "masks": np.empty((0, height, width), dtype=bool),
            }

        boxes = np.stack(boxes).astype(np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        class_ids = np.asarray(class_ids, dtype=np.int64)
        masks = np.stack(masks).astype(bool)

        boxes, scores, class_ids, masks = self._classwise_nms(
            boxes, scores, class_ids, masks, nms_thr
        )

        if len(scores) > max_num_boxes:
            order = np.argsort(-scores)[:max_num_boxes]
            boxes, scores, class_ids, masks = (
                boxes[order],
                scores[order],
                class_ids[order],
                masks[order],
            )

        print(
            "[SAM3 DEBUG] after score/NMS: "
            f"{len(scores)} detections, scores: "
            f"{sorted([round(float(s), 3) for s in scores], reverse=True)[:10]}"
        )
        return {
            "boxes": boxes,
            "scores": scores,
            "class_ids": class_ids,
            "masks": masks,
        }

    def __call__(self, input_image, max_num_boxes=100, score_thr=None, nms_thr=0.5):
        results = self.detect(input_image, max_num_boxes, score_thr, nms_thr)
        return (
            (results["boxes"],),
            results["scores"],
            results["class_ids"],
            results["masks"],
        )


class YOLOW():

    def __init__(self,YOLOW_PATH):
        from ultralytics import YOLO

        self.model_name = os.environ.get("EMPOWER_YOLOW_MODEL", "yolov8l-worldv2.pt")
        os.makedirs(YOLOW_PATH, exist_ok=True)
        if os.path.isabs(self.model_name) or os.path.dirname(self.model_name):
            self.model = YOLO(self.model_name)
        else:
            cwd = os.getcwd()
            try:
                os.chdir(YOLOW_PATH)
                self.model = YOLO(self.model_name)
            finally:
                os.chdir(cwd)
        self.objects = []
        self.class_names = ""

    def _sync_clip_model_device(self):
        world_model = getattr(self.model, "model", None)
        clip_model = getattr(world_model, "clip_model", None)
        model_layers = getattr(world_model, "model", None)
        if clip_model is None or model_layers is None:
            return

        try:
            device = next(model_layers.parameters()).device
        except StopIteration:
            return

        clip_model.to(device)
        if hasattr(clip_model, "device"):
            clip_model.device = device

    def set_class_name(self, objects):
        if isinstance(objects, (list, tuple)):
            self.objects = [obj.strip() for obj in objects if obj and obj.strip()]
            self.class_names = ",".join(self.objects)
        else:
            self.class_names = objects
            self.objects = [obj.strip() for obj in objects.split(",") if obj.strip()]
        if not self.objects:
            return
        # The extra background prompt mirrors the original YOLO-World wrapper,
        # while get_class_name() filters it out of Empower's object mapping.
        self._sync_clip_model_device()
        self.model.set_classes(self.objects + [" "])

    def get_class_name(self, id):
        id = int(id)
        if id < 0 or id >= len(self.objects):
            return None
        return self.objects[id]

    def __call__(self,input_image,max_num_boxes=100,score_thr=0.05,nms_thr=0.5):
        if not self.objects:
            return (np.empty((0, 4), dtype=np.float32),), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)

        results = self.model.predict(
            input_image,
            conf=score_thr,
            iou=nms_thr,
            max_det=max_num_boxes,
            verbose=False,
        )
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            print("[YOLOW DEBUG] after score/NMS: 0 detections")
            return (np.empty((0, 4), dtype=np.float32),), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)

        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        confidence = boxes.conf.detach().cpu().numpy().astype(np.float32)
        class_id = boxes.cls.detach().cpu().numpy().astype(np.int64)

        valid = class_id < len(self.objects)
        xyxy = xyxy[valid]
        confidence = confidence[valid]
        class_id = class_id[valid]

        print(f"[YOLOW DEBUG] after score/NMS: {len(confidence)} detections, scores: {sorted([round(float(s),3) for s in confidence], reverse=True)[:10]}")
        return (xyxy,), confidence, class_id
