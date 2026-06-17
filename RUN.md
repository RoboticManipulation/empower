## Semantic placement via Empower 


### Refined semantic placement

This runs the refined mode in `semantic_placement_refined`. It asks the LLM for
one visible semantic reference, grounds that reference, and returns a
camera-frame coordinate using the reference centroid plus the refined offset.

#### Refined with SAM3

```bash
python3 ~/ws/packages/empower/src/visualize_semantic_placement.py \
  ~/ws/packages/geo_sem_place_dataset/scenes/real_world/orbbec_gemini_336/place/7/rgb_0.png \
  ~/ws/packages/geo_sem_place_dataset/scenes/real_world/orbbec_gemini_336/place/7/pc_0.pcd \
  --grasp-object "ketchup bottle" \
  --camera-info ~/ws/packages/geo_sem_place_dataset/scenes/real_world/orbbec_gemini_336/camera_intrinsics.json \
  --detector-backend sam3 \
  --semantic-mode semantic_placement_refined \
  --relation-offset-m 0.15 \
  --no-window \
  --write-prefix ~/ws/packages/empower/output/semantic_placement_compare/ketchup_rgb0_refined_sam3
```

#### Refined with YOLO-World

```bash
python3 ~/ws/packages/empower/src/visualize_semantic_placement.py \
  ~/ws/packages/geo_sem_place_dataset/scenes/real_world/orbbec_gemini_336/place/7/rgb_3.png \
  ~/ws/packages/geo_sem_place_dataset/scenes/real_world/orbbec_gemini_336/place/7/pc_3.pcd \
  --grasp-object "ketchup bottle" \
  --camera-info ~/ws/packages/geo_sem_place_dataset/scenes/real_world/orbbec_gemini_336/camera_intrinsics.json \
  --detector-backend yolow \
  --semantic-mode semantic_placement_refined \
  --relation-offset-m 0.15 \
  --no-window \
  --write-prefix ~/ws/packages/empower/output/semantic_placement_compare/ketchup_rgb3_refined_yolow
```

### Baseline semantic placement

This runs the baseline mode in `semantic_placement`. It keeps the LLM/action
style closer to the original planner, grounds the planned reference object
centroid, applies original-style relation offsets, and returns the camera-frame
coordinate instead of sending it to MoveIt.

#### Baseline with SAM3

```bash
python3 ~/ws/packages/empower/src/visualize_semantic_placement.py \
  ~/ws/packages/geo_sem_place_dataset/scenes/real_world/orbbec_gemini_336/place/7/rgb_0.png \
  ~/ws/packages/geo_sem_place_dataset/scenes/real_world/orbbec_gemini_336/place/7/pc_0.pcd \
  --grasp-object "ketchup bottle" \
  --camera-info ~/ws/packages/geo_sem_place_dataset/scenes/real_world/orbbec_gemini_336/camera_intrinsics.json \
  --detector-backend sam3 \
  --semantic-mode semantic_placement \
  --relation-offset-m 0.15 \
  --no-window \
  --write-prefix ~/ws/packages/empower/output/semantic_placement_compare/ketchup_rgb3_baseline_sam3
```

#### Baseline with YOLO-World

```bash
python3 ~/ws/packages/empower/src/visualize_semantic_placement.py \
  ~/ws/packages/geo_sem_place_dataset/scenes/real_world/orbbec_gemini_336/place/7/rgb_3.png \
  ~/ws/packages/geo_sem_place_dataset/scenes/real_world/orbbec_gemini_336/place/7/pc_3.pcd \
  --grasp-object "ketchup bottle" \
  --camera-info ~/ws/packages/geo_sem_place_dataset/scenes/real_world/orbbec_gemini_336/camera_intrinsics.json \
  --detector-backend yolow \
  --semantic-mode semantic_placement \
  --relation-offset-m 0.15 \
  --no-window \
  --write-prefix ~/ws/packages/empower/output/semantic_placement_compare/ketchup_rgb3_baseline_yolow
```


For the two detector setups:

  - SAM3: confidence threshold is 0.3
      - Env override: EMPOWER_SAM3_SCORE_THR
      - Code: src/detection.py:472
  - YOLO-world: confidence threshold is 0.05
      - Env override: EMPOWER_YOLOW_SCORE_THR
      - Code: src/detection.py:492

----
 Current formula:

  reference = centroid(reference object pointcloud)
  offset = value passed by --relation-offset-m
      

  left:
    coordinate = [reference_x - offset, reference_y, reference_z]

  right:
    coordinate = [reference_x + offset, reference_y, reference_z]





# IGNORE THE BELOW


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

