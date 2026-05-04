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

Edit `configs/llm_config.yaml` and set `llm_provider` (`"openai"` or `"mixtral"`), then export the matching API key:

```bash
export OPENAI_API_KEY=<YOUR API KEY>
# or
export MISTRAL_API_KEY=<YOUR API KEY>
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
