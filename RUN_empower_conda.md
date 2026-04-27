# How to run Empower 


---

## Step 1 — Create conda env and download weights

```bash
bash set_up_empower
```

This creates the `empower` conda environment (if missing), installs Python dependencies, and fetches model files under `config/`.

---

## Step 2 — LLM API keys

Edit `configs/llm_config.yaml` and set `llm_provider` (`"openai"` or `"mixtral"`), then either:

- set `openai_api_key` / `mistral_api_key` in that file, or  
- leave them as `""` and export **`OPENAI_API_KEY`** or **`MISTRAL_API_KEY`** in your shell (see comments in that YAML).

---

## Step 3 — Prepare local data

```bash
cd src
conda activate empower

python3 prepare_local_data.py order_by_height 5 0
python3 prepare_local_data.py order_by_height 5 1
```

---

## Step 4 — Load models (cache)

```bash
cd src
conda activate empower
python3 models_cacher.py order_by_height
```

---

## Step 5 — Run detection / task

```bash
cd src
conda activate empower
python3 execute_task.py
```


## Step 6 — 

```bash
USE_CASE=order_by_height python3 color_pcl_local.py
```