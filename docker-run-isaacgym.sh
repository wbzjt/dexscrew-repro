#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/Codefield/py/dexscrew-repro}"
ISAACGYM_DIR="${ISAACGYM_DIR:-$HOME/isaacgym}"
USER_ID="${USER_ID:-$(id -u)}"
GROUP_ID="${GROUP_ID:-$(id -g)}"

docker run --rm -it \
  --name dexscrew-dev \
  --gpus all \
  --runtime=nvidia \
  --user "${USER_ID}:${GROUP_ID}" \
  --ipc=host \
  --network host \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e HOME=/tmp \
  -e TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  -v "${PROJECT_DIR}:/workspace/dexscrew-repro" \
  -v "${ISAACGYM_DIR}:/opt/isaacgym" \
  -w /workspace/dexscrew-repro \
  dexscrew:ig20-py38 \
  bash
