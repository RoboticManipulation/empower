
import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
from efficientvit.export_encoder import SamResize
from efficientvit.inference import SamDecoder, SamEncoder
from ultralytics import YOLO

class VitSam():

    def __init__(self, encoder_model, decoder_model):
        self.decoder = SamDecoder(decoder_model)
        self.encoder = SamEncoder(encoder_model)


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
        resize_transform = SamResize(img_size)
        x = resize_transform(x).float() / 255
        x = transforms.Normalize(mean=pixel_mean, std=pixel_std)(x)

        h, w = x.shape[-2:]
        th, tw = img_size, img_size
        assert th >= h and tw >= w
        x = F.pad(x, (0, tw - w, 0, th - h), value=0).unsqueeze(0).numpy()

        return x
    
    

class YOLOW():

    def __init__(self,YOLOW_PATH):
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
