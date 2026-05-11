#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?usage: paper_codrive_supervisor.sh RUN_DIR [main_session] [post_session]}"
MAIN_SESSION="${2:-}"
POST_SESSION="${3:-}"
INTERVAL_SEC="${INTERVAL_SEC:-60}"

mkdir -p "${RUN_DIR}/status"
STATUS_FILE="${RUN_DIR}/status/supervisor_status.txt"
EVENT_LOG="${RUN_DIR}/status/supervisor_events.log"
LAST_PHASE_FILE="${RUN_DIR}/status/.supervisor_last_phase"
RUN_NAME="$(basename "${RUN_DIR}")"

emit_event() {
  printf '%s\t%s\n' "$(date -Iseconds)" "$*" >> "${EVENT_LOG}"
}

count_bad_patterns() {
  { grep -RInE \
    'Traceback|RuntimeError|CUDA out of memory|Error executing job|Missing key\(s\)|Unexpected key\(s\)|FileNotFoundError|AssertionError' \
    "${RUN_DIR}/logs" 2>/dev/null || true; } | wc -l
}

count_matching_proc() {
  local base_pattern="$1"
  { pgrep -af "${base_pattern}" 2>/dev/null || true; } | awk -v run="${RUN_NAME}" 'index($0, run) > 0 && $2 ~ /python/ { n++ } END { print n + 0 }'
}

gpu_csv() {
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "NA,NA,NA,NA,NA"
}

session_ok() {
  local session="$1"
  [[ -z "${session}" ]] && return 0
  tmux has-session -t "${session}" 2>/dev/null
}

while true; do
  now="$(date -Iseconds)"
  phase="$(cat "${RUN_DIR}/status/phase.txt" 2>/dev/null || echo missing_phase)"
  previous_phase="$(cat "${LAST_PHASE_FILE}" 2>/dev/null || true)"
  if [[ "${phase}" != "${previous_phase}" ]]; then
    emit_event "phase_change ${previous_phase:-none} -> ${phase}"
    printf '%s\n' "${phase}" > "${LAST_PHASE_FILE}"
  fi

  gpu_line="$(gpu_csv)"
  gpu_mem_used="$(printf '%s\n' "${gpu_line}" | awk -F, '{gsub(/ /,"",$2); print $2}')"
  gpu_util="$(printf '%s\n' "${gpu_line}" | awk -F, '{gsub(/ /,"",$4); print $4}')"
  bad_count="$(count_bad_patterns | tr -d ' ')"
  train_proc_count="$(count_matching_proc 'python train.py' | tr -d ' ')"
  eval_proc_count="$(count_matching_proc 'paper_codrive_eval.py' | tr -d ' ')"
  deploy_ckpts="$(find "${RUN_DIR}/train_outputs" \( -type f -o -type l \) -name 'model_best_deploy.ckpt' 2>/dev/null | wc -l | tr -d ' ')"
  status_files="$(find "${RUN_DIR}/status" -maxdepth 1 -type f -name '*.status' 2>/dev/null | wc -l | tr -d ' ')"
  alert="ok"

  if ! session_ok "${MAIN_SESSION}" && [[ "${phase}" != "done" ]]; then
    alert="main_tmux_missing"
  elif [[ "${bad_count}" != "0" ]]; then
    alert="bad_log_patterns"
  elif [[ "${phase}" == "formal_training" || "${phase}" == "representation_train" ]]; then
    if [[ "${train_proc_count}" == "0" ]]; then
      alert="training_phase_no_train_process"
    elif [[ "${gpu_util}" != "NA" && "${gpu_util}" -lt 5 ]]; then
      alert="training_phase_low_gpu_util"
    fi
  elif [[ "${phase}" == *"eval"* || "${phase}" == "robustness" || "${phase}" == "nfe_latency" ]]; then
    if [[ "${eval_proc_count}" == "0" && "${train_proc_count}" == "0" && "${phase}" != "done" ]]; then
      alert="eval_phase_no_worker_process"
    fi
  fi

  {
    echo "time=${now}"
    echo "run_dir=${RUN_DIR}"
    echo "run_name=${RUN_NAME}"
    echo "phase=${phase}"
    echo "main_session=${MAIN_SESSION}"
    echo "post_session=${POST_SESSION}"
    echo "main_session_ok=$(session_ok "${MAIN_SESSION}" && echo true || echo false)"
    echo "post_session_ok=$(session_ok "${POST_SESSION}" && echo true || echo false)"
    echo "gpu_csv=${gpu_line}"
    echo "gpu_mem_used_mib=${gpu_mem_used}"
    echo "gpu_util_pct=${gpu_util}"
    echo "train_proc_count=${train_proc_count}"
    echo "eval_proc_count=${eval_proc_count}"
    echo "bad_log_pattern_count=${bad_count}"
    echo "status_file_count=${status_files}"
    echo "deploy_ckpt_count=${deploy_ckpts}"
    echo "alert=${alert}"
  } > "${STATUS_FILE}"

  if [[ "${alert}" != "ok" ]]; then
    emit_event "ALERT ${alert} phase=${phase} train=${train_proc_count} eval=${eval_proc_count} bad=${bad_count} gpu=${gpu_line}"
  fi

  sleep "${INTERVAL_SEC}"
done
