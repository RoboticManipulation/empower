# How to run Empower 


---

## Step 1 — Install in the devcontainer Python and download weights

```bash
cd /home/$USER/ws/packages/empower
./set_up_empower
```

This uses the container's default Python 3.10 and keeps the devcontainer Torch stack intact. Detection runs through SAM3, with the gated `facebook/sam3` checkpoint resolved through Hugging Face or `EMPOWER_SAM3_CHECKPOINT`. It does not use conda or a separate Python virtual environment.

Useful rerun options:

```bash
EMPOWER_INSTALL_DEPS=0 ./set_up_empower       # only create folders/download missing weights
EMPOWER_DOWNLOAD_WEIGHTS=0 ./set_up_empower  # only install/verify Python dependencies
EMPOWER_SAM3_CHECKPOINT=/path/to/sam3.pt ./set_up_empower  # optional local SAM3 checkpoint
```

---

## Step 2 — LLM provider and API key

Edit `configs/llm_config.yaml` and set `llm_provider` (`"openai"`, `"mixtral"`, or `"openrouter"`), then export the matching API key:

```bash
export OPENAI_API_KEY=<YOUR API KEY>
# or
export MISTRAL_API_KEY=<YOUR API KEY>
# or
export OPENROUTER_API_KEY=<YOUR API KEY>
```

---

## Step 3 — Prepare local data

```bash
cd src

python3 prepare_local_data.py order_by_height 5 0
python3 prepare_local_data.py order_by_height 5 1
```

---

## Step 4 — Load models (cache)

```bash
cd src
python3 models_cacher.py order_by_height
```

---

## Step 5 — Run detection / task

```bash
cd src
python3 execute_task.py
```


## Step 6 — 

```bash
cd src
USE_CASE=order_by_height python3 color_pcl_local.py
```


## Whole pipeline in one go

Example with SAM3 (default):

```bash
bash -ic 'python3 src/visualize_semantic_placement.py \
  geo_sem_place_dataset/scenes/real_world/orbbec_gemini_336/place/7/rgb_0.png \
  geo_sem_place_dataset/scenes/real_world/orbbec_gemini_336/place/7/pc_0.pcd \
  --grasp-object "ketchup bottle" \
  --camera-info geo_sem_place_dataset/scenes/real_world/orbbec_gemini_336/camera_intrinsics.json \
  --detector-backend sam3 \
  --no-window \
  --write-prefix output/semantic_placement_demo/ketchup_bottle_sam3'
```

Example with YOLO-World:

```bash
bash -ic 'python3 src/visualize_semantic_placement.py \
  geo_sem_place_dataset/scenes/real_world/orbbec_gemini_336/place/7/rgb_0.png \
  geo_sem_place_dataset/scenes/real_world/orbbec_gemini_336/place/7/pc_0.pcd \
  --grasp-object "ketchup bottle" \
  --camera-info geo_sem_place_dataset/scenes/real_world/orbbec_gemini_336/camera_intrinsics.json \
  --detector-backend yolow \
  --no-window \
  --write-prefix output/semantic_placement_demo/ketchup_bottle_yolow'
```


For the two detector setups:

  - SAM3: confidence threshold is 0.3
      - Env override: EMPOWER_SAM3_SCORE_THR
      - Code: src/detection.py:454
  - YOLO-world: confidence threshold is 0.05
      - Env override: EMPOWER_YOLOW_SCORE_THR
      - Code: src/detection.py:474

----
 Current formula:

  reference = centroid(reference object pointcloud)

  left:
    coordinate = [reference_x - 0.15, reference_y, reference_z]

  right:
    coordinate = [reference_x + 0.15, reference_y, reference_z]