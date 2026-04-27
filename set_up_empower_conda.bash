#!/usr/bin/env bash
# Standalone setup for Empower (no devcontainer required).
# - Creates the "empower" conda env and installs Python deps only if the env does not exist.
# - Always ensures model weights under config/ (skips any file that is already non-empty).
# - Creates images/ and output/ at the repo root.
#
# Usage: from the empower repo root:
#   chmod +x set_up_empower   # once
#   ./set_up_empower
#
# Optional: MINICONDA=/path/to/miniconda3 if conda is not on PATH and not under ~/miniconda3 or ~/anaconda3.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="empower"
PYTHON_VER="3.8.18"

err() { echo "[ERROR] $*" >&2; exit 1; }

_source_conda() {
  if [ -n "${MINICONDA:-}" ] && [ -f "${MINICONDA}/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "${MINICONDA}/etc/profile.d/conda.sh"
    return 0
  fi
  if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    return 0
  fi
  if [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    return 0
  fi
  if [ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "${HOME}/miniforge3/etc/profile.d/conda.sh"
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    local base
    base="$(conda info --base)"
    # shellcheck source=/dev/null
    source "${base}/etc/profile.d/conda.sh"
    return 0
  fi
  err "conda not found. Install Miniconda/Miniforge, or set MINICONDA to the install prefix."
}

# Resume-friendly download (same URLs as .devcontainer/noetic-empower/postCreate.sh).
_download_if_missing() {
  local url="$1"
  local dest="$2"
  if [ -s "${dest}" ]; then
    echo "[INFO] Already present — skip: ${dest}"
    return 0
  fi
  echo "[INFO] Downloading → ${dest}"
  mkdir -p "$(dirname "${dest}")"
  curl -L --fail --retry 5 --retry-delay 5 -C - -o "${dest}.part" "${url}"
  mv "${dest}.part" "${dest}"
  ls -lah "${dest}"
}

_download_model_weights() {
  command -v curl >/dev/null 2>&1 || err "curl is required for weight downloads."

  echo ""
  echo "[INFO] Model weights (YOLO-World + EfficientViT-SAM L2)…"

  _download_if_missing \
    "https://huggingface.co/wondervictor/YOLO-World-V2.1/resolve/main/l_stage1-7d280586.pth" \
    "${ROOT}/config/yolow/yolow.pth"

  _download_if_missing \
    "https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/onnx/l2_encoder.onnx" \
    "${ROOT}/config/efficientvitsam/l2_encoder.onnx"

  _download_if_missing \
    "https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/onnx/l2_decoder.onnx" \
    "${ROOT}/config/efficientvitsam/l2_decoder.onnx"

  _download_if_missing \
    "https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l2.pt" \
    "${ROOT}/config/efficientvitsam/efficientvit_sam_l2.pt"
}

_source_conda
CONDA_BASE="$(conda info --base)"
ENV_PREFIX="${CONDA_BASE}/envs/${ENV_NAME}"

if [ -d "${ENV_PREFIX}" ]; then
  echo "[INFO] Conda env '${ENV_NAME}' already exists at ${ENV_PREFIX}"
  echo "[INFO] Skipping conda create and pip/mim install."
  echo "[INFO] To reinstall from scratch: conda remove -n ${ENV_NAME} --all -y  then re-run this script."
else
  echo "[INFO] Creating conda env '${ENV_NAME}' (Python ${PYTHON_VER})..."
  conda create -n "${ENV_NAME}" "python=${PYTHON_VER}" -y

  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"

  cd "${ROOT}"
  if [ ! -f requirements.txt ]; then
    err "requirements.txt not found in ${ROOT}"
  fi

  echo "[INFO] Installing Python dependencies (several minutes; downloads PyTorch, mmcv, etc.)..."
  pip install --upgrade pip
  pip install "spacy<3.8" "numpy<1.25"
  pip install -r requirements.txt
  pip install -U openmim
  mim install mmcv==2.0.0
  mim install mmyolo mmdet
  python -m spacy download en_core_web_sm
  pip install tomli
fi

mkdir -p "${ROOT}/images" "${ROOT}/output"
_download_model_weights

echo ""
echo "[OK] Empower repo is ready at ${ROOT}"
echo "  conda activate ${ENV_NAME}"
echo "  cd ${ROOT}/src"
echo ""
echo "[NOTE] Set MISTRAL_API_KEY or OPENAI_API_KEY (see configs/llm_config.yaml)."