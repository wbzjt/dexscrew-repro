#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_ID="${RUN_ID:-m24_indexrelaxed_j1_105_s42_20260804_212706_ppo8192}"
STAGE_DIR="${PROJECT_DIR}/outputs/XHandPasiniM24NutBolt_teacher/${RUN_ID}/stage1_nn"
ISAACGYM_DIR="${ISAACGYM_DIR:-/data/Codefield/third_party/isaacgym_preview4_py38_clean}"
CONTAINER_NAME="${DEXSCREW_CONTAINER_NAME:-dexscrew_m24_ppo_viewer}"

if [[ -z "${DISPLAY:-}" ]]; then
  echo "DISPLAY is empty. Run this command from the Ubuntu desktop terminal." >&2
  exit 1
fi

CHECKPOINT="$({
  find "${STAGE_DIR}" -maxdepth 1 -type f -name 'best_reward_*.pth' \
    -printf '%T@ %p\n' 2>/dev/null || true
} | sort -nr | head -n 1 | cut -d' ' -f2-)"

if [[ -z "${CHECKPOINT}" ]] || [[ ! -f "${CHECKPOINT}" ]]; then
  echo "No PPO checkpoint found under: ${STAGE_DIR}" >&2
  exit 1
fi

SNAPSHOT_DIR="${STAGE_DIR}/eval_snapshots"
mkdir -p "${SNAPSHOT_DIR}"
SNAPSHOT="${SNAPSHOT_DIR}/$(date +%Y%m%d_%H%M%S)_$(basename "${CHECKPOINT}")"
cp -- "${CHECKPOINT}" "${SNAPSHOT}"
CHECKPOINT="${SNAPSHOT}"

CHECKPOINT="${CHECKPOINT#${PROJECT_DIR}/}"
echo "Loading M24 PPO evaluation snapshot: ${CHECKPOINT}"

cd "${PROJECT_DIR}"
ISAACGYM_DIR="${ISAACGYM_DIR}" \
DEXSCREW_CONTAINER_NAME="${CONTAINER_NAME}" \
  exec "${PROJECT_DIR}/docker-run-isaacgym.sh" python -u train.py \
    task=XHandPasiniM24NutBolt \
    headless=False \
    test=True \
    seed=42 \
    experiment=rl \
    train.algo=PPO \
    task.env.numEnvs=1 \
    sim_device=cuda:0 \
    rl_device=cuda:0 \
    graphics_device_id=0 \
    wandb_activate=False \
    "checkpoint=${CHECKPOINT}"
