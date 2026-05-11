#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?usage: paper_codrive_active_guard.sh RUN_DIR [main_session] [post_session] [aliaser_session]}"
MAIN_SESSION="${2:-}"
POST_SESSION="${3:-}"
ALIASER_SESSION="${4:-}"
INTERVAL_SEC="${INTERVAL_SEC:-300}"

RUN_NAME="$(basename "${RUN_DIR}")"
STATUS_DIR="${RUN_DIR}/status"
LOG_DIR="${RUN_DIR}/logs"
TRAIN_DIR="${RUN_DIR}/train_outputs"

mkdir -p "${STATUS_DIR}"

STATUS_FILE="${STATUS_DIR}/active_guard_status.txt"
EVENT_LOG="${STATUS_DIR}/active_guard_events.log"
PROGRESS_FILE="${STATUS_DIR}/training_progress.tsv"
CONTROL_FILE="${STATUS_DIR}/CONTROL_REQUIRED.txt"
LAST_PHASE_FILE="${STATUS_DIR}/.active_guard_last_phase"

methods=(padapt purebc diffusion_latent consistency_latent flow_matching diffusion_action_chunk_len1)
seeds=(42 43 44)

emit_event() {
  printf '%s\t%s\n' "$(date -Iseconds)" "$*" >> "${EVENT_LOG}"
}

session_ok() {
  local session="$1"
  [[ -z "${session}" ]] && return 0
  tmux has-session -t "${session}" 2>/dev/null
}

count_matching_proc() {
  local base_pattern="$1"
  { pgrep -af "${base_pattern}" 2>/dev/null || true; } | awk -v run="${RUN_NAME}" 'index($0, run) > 0 && $2 ~ /python/ { n++ } END { print n + 0 }'
}

bad_pattern_count() {
  { grep -RInE 'Traceback|RuntimeError|CUDA out of memory|Error executing job|Missing key\(s\)|Unexpected key\(s\)|FileNotFoundError|AssertionError' "${LOG_DIR}" 2>/dev/null || true; } | wc -l | tr -d ' '
}

gpu_line() {
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "NA,NA,NA,NA,NA"
}

nn_dir_for_method() {
  case "$1" in
    padapt) echo "stage2_nn" ;;
    purebc) echo "stage2_bc_nn" ;;
    diffusion_latent) echo "stage2_diffusion_nn" ;;
    consistency_latent) echo "stage2_consistency_nn" ;;
    flow_matching) echo "stage2_flow_nn" ;;
    diffusion_action_chunk_len1) echo "stage2_diffusion_action_chunk_nn" ;;
    *) echo "stage2_nn" ;;
  esac
}

safe_alias_train_best() {
  while IFS= read -r -d '' train_ckpt; do
    local dir deploy
    dir="$(dirname "${train_ckpt}")"
    deploy="${dir}/model_best_deploy.ckpt"
    if [[ ! -e "${deploy}" && ! -L "${deploy}" ]]; then
      (cd "${dir}" && ln -s model_best_train.ckpt model_best_deploy.ckpt)
      printf '%s\t%s\t%s\tcreated_by_active_guard\n' "$(date -Iseconds)" "${train_ckpt}" "${deploy}" >> "${STATUS_DIR}/checkpoint_aliases.tsv"
      emit_event "created_checkpoint_alias ${deploy}"
    fi
  done < <(find "${TRAIN_DIR}" -type f -name 'model_best_train.ckpt' -print0 2>/dev/null || true)
}

latest_best() {
  local log="$1"
  [[ -f "${log}" ]] || { echo ""; return; }
  tail -c 200000 "${log}" | tr '\r' '\n' | grep -o 'Current Best: [0-9.]*' | tail -n 1 | awk '{print $3}'
}

write_progress_snapshot() {
  printf 'time\tmethod\tseed\tstatus\tbest\ttrain_ckpt\tdeploy_ckpt\n' > "${PROGRESS_FILE}"
  local now="$1"
  local method seed status best nn train_ckpt deploy_ckpt
  for method in "${methods[@]}"; do
    for seed in "${seeds[@]}"; do
      [[ "${method}" == "diffusion_action_chunk_len1" && "${seed}" != "42" ]] && continue
      status="$(cat "${STATUS_DIR}/train_${method}_s${seed}.status" 2>/dev/null || echo running_or_pending)"
      best="$(latest_best "${LOG_DIR}/train_${method}_s${seed}.log")"
      nn="$(nn_dir_for_method "${method}")"
      train_ckpt="${TRAIN_DIR}/${method}_s${seed}/${nn}/model_best_train.ckpt"
      deploy_ckpt="${TRAIN_DIR}/${method}_s${seed}/${nn}/model_best_deploy.ckpt"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${now}" "${method}" "${seed}" "${status}" "${best}" "$([[ -e "${train_ckpt}" ]] && echo yes || echo no)" "$([[ -e "${deploy_ckpt}" || -L "${deploy_ckpt}" ]] && echo yes || echo no)" >> "${PROGRESS_FILE}"
    done
  done
}

while true; do
  now="$(date -Iseconds)"
  phase="$(cat "${STATUS_DIR}/phase.txt" 2>/dev/null || echo missing_phase)"
  previous_phase="$(cat "${LAST_PHASE_FILE}" 2>/dev/null || true)"
  if [[ "${phase}" != "${previous_phase}" ]]; then
    emit_event "phase_change ${previous_phase:-none} -> ${phase}"
    printf '%s\n' "${phase}" > "${LAST_PHASE_FILE}"
  fi

  safe_alias_train_best
  write_progress_snapshot "${now}"

  gpu="$(gpu_line)"
  gpu_util="$(printf '%s\n' "${gpu}" | awk -F, '{gsub(/ /,"",$4); print $4}')"
  train_proc_count="$(count_matching_proc 'python train.py' | tr -d ' ')"
  eval_proc_count="$(count_matching_proc 'paper_codrive_eval.py' | tr -d ' ')"
  train_status_count="$(find "${STATUS_DIR}" -maxdepth 1 -type f -name 'train_*.status' 2>/dev/null | wc -l | tr -d ' ')"
  eval_json_count="$(find "${RUN_DIR}/raw_json" -type f -name '*.json' 2>/dev/null | wc -l | tr -d ' ')"
  deploy_count="$(find "${TRAIN_DIR}" \( -type f -o -type l \) -name 'model_best_deploy.ckpt' 2>/dev/null | wc -l | tr -d ' ')"
  bad_count="$(bad_pattern_count)"
  alert="ok"
  recommendation="continue"

  if [[ "${bad_count}" != "0" ]]; then
    alert="critical_bad_log_pattern"
    recommendation="inspect_latest_error_and_stop_or_patch_before_more_eval"
  elif ! session_ok "${MAIN_SESSION}" && [[ "${phase}" != "done" ]]; then
    alert="critical_main_session_missing"
    recommendation="inspect_tmux_and_resume_or_mark_aborted"
  elif [[ "${phase}" == "formal_training" && "${train_proc_count}" == "0" && "${train_status_count}" -lt 15 ]]; then
    alert="critical_training_phase_no_train_process"
    recommendation="inspect_jobs_and_restart_missing_train_batch_if_safe"
  elif [[ "${phase}" == *"eval"* && "${eval_proc_count}" == "0" && "${phase}" != "done" ]]; then
    alert="warning_eval_phase_no_eval_process"
    recommendation="check_whether_eval_is_between_jobs_or_stalled"
  elif [[ "${phase}" == "formal_training" && "${gpu_util}" != "NA" && "${gpu_util}" -lt 10 && "${train_proc_count}" != "0" ]]; then
    alert="warning_low_gpu_util_with_train_process"
    recommendation="inspect_logs_for_slow_or_blocked_sim"
  fi

  if [[ "${alert}" != "ok" ]]; then
    {
      echo "time=${now}"
      echo "alert=${alert}"
      echo "recommendation=${recommendation}"
      echo "phase=${phase}"
      echo "train_proc_count=${train_proc_count}"
      echo "eval_proc_count=${eval_proc_count}"
      echo "bad_log_pattern_count=${bad_count}"
      echo "gpu_csv=${gpu}"
    } > "${CONTROL_FILE}"
    emit_event "ALERT ${alert} recommendation=${recommendation}"
  else
    rm -f "${CONTROL_FILE}"
  fi

  {
    echo "time=${now}"
    echo "run_dir=${RUN_DIR}"
    echo "phase=${phase}"
    echo "main_session_ok=$(session_ok "${MAIN_SESSION}" && echo true || echo false)"
    echo "post_session_ok=$(session_ok "${POST_SESSION}" && echo true || echo false)"
    echo "aliaser_session_ok=$(session_ok "${ALIASER_SESSION}" && echo true || echo false)"
    echo "gpu_csv=${gpu}"
    echo "train_proc_count=${train_proc_count}"
    echo "eval_proc_count=${eval_proc_count}"
    echo "train_status_count=${train_status_count}"
    echo "eval_json_count=${eval_json_count}"
    echo "deploy_ckpt_count=${deploy_count}"
    echo "bad_log_pattern_count=${bad_count}"
    echo "alert=${alert}"
    echo "recommendation=${recommendation}"
    echo "progress_file=${PROGRESS_FILE}"
    echo "control_file=${CONTROL_FILE}"
  } > "${STATUS_FILE}"

  [[ "${phase}" == "done" ]] && exit 0
  sleep "${INTERVAL_SEC}"
done
