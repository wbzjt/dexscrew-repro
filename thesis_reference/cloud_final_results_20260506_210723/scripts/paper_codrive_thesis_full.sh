#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/code/dexscrew-repro}"
cd "${ROOT}"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /root/miniconda3/etc/profile.d/conda.sh
elif [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /opt/conda/etc/profile.d/conda.sh
fi
conda activate dexscrew-ig

export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1
export DEXSCREW_SKIP_GIT_DIFF=1

GPU="${GPU:-0}"
PARALLEL_TRAIN="${PARALLEL_TRAIN:-3}"
WINDOW_SEC="${WINDOW_SEC:-18000}"
TASK="${TASK:-Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis}"
TEACHER_CKPT="${TEACHER_CKPT:-sim2real/codrive_thesis/best_reward_3655.17.pth}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-paper_codrive_thesis_full_${RUN_ID}}"
RUN_DIR="${RUN_DIR:-outputs/${RUN_NAME}}"

mkdir -p "${RUN_DIR}"/{logs,status,raw_json,raw_csv,aggregate_csv,figures,train_outputs,eval_tmp}
ln -sfn "${RUN_NAME}" outputs/paper_codrive_thesis_full_latest
PHASE_FILE="${RUN_DIR}/status/phase.txt"
COMMANDS_LOG="${RUN_DIR}/commands.log"
JOBS_TSV="${RUN_DIR}/jobs.tsv"

touch "${COMMANDS_LOG}"
printf 'phase\tstatus\ttime\n' > "${JOBS_TSV}"
printf 'starting\n' > "${PHASE_FILE}"

log_phase() {
  local phase="$1"
  printf '%s\n' "${phase}" > "${PHASE_FILE}"
  printf '%s\t%s\t%s\n' "${phase}" "running" "$(date -Iseconds)" >> "${JOBS_TSV}"
}

finish_phase() {
  local phase="$1"
  local status="$2"
  printf '%s\t%s\t%s\n' "${phase}" "${status}" "$(date -Iseconds)" >> "${JOBS_TSV}"
}

qcmd() {
  printf '%q ' "$@"
  printf '\n'
}

write_manifest() {
  {
    echo "run_name=${RUN_NAME}"
    echo "run_dir=${RUN_DIR}"
    echo "root=${ROOT}"
    echo "host=$(hostname)"
    echo "start_time=$(date -Iseconds)"
    echo "gpu=${GPU}"
    echo "parallel_train=${PARALLEL_TRAIN}"
    echo "window_sec=${WINDOW_SEC}"
    echo "task=${TASK}"
    echo "teacher_ckpt=${TEACHER_CKPT}"
    echo
    echo "[git]"
    git rev-parse --abbrev-ref HEAD || true
    git rev-parse HEAD || true
    git status --short || true
    echo
    echo "[sha256]"
    sha256sum "configs/task/${TASK}.yaml" "configs/train/${TASK}.yaml" "${TEACHER_CKPT}" || true
    echo
    echo "[gpu]"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader || true
  } > "${RUN_DIR}/manifest.txt"
}

monitor_gpu() {
  while true; do
    printf '%s\t' "$(date -Iseconds)"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits || true
    sleep 30
  done
}

common_overrides() {
  printf '%s\n' \
    "task=${TASK}" \
    "headless=True" \
    "sim_device=cuda:${GPU}" \
    "rl_device=cuda:${GPU}" \
    "graphics_device_id=7" \
    "task.env.termination.grace_steps=0" \
    "task.env.termination.enable_finger_dist=True" \
    "task.env.termination.enable_nut_stagnation=True" \
    "task.env.termination.enable_no_contact=True" \
    "task.env.termination.enable_screw_limit=True" \
    "task.env.termination.log=True" \
    "task.env.randomization.obs_noise_t_scale=0.01" \
    "task.env.randomization.obs_noise_e_scale=0.02" \
    "task.env.numEnvs=48" \
    "train.ppo.minibatch_size=576" \
    "wandb_activate=False"
}

eval_select_overrides() {
  printf '%s\n' \
    "train.ppo.eval_select.enabled=True" \
    "train.ppo.eval_select.interval_agent_steps=20000000" \
    "train.ppo.eval_select.min_agent_steps=20000000" \
    "train.ppo.eval_select.num_steps=512" \
    "train.ppo.eval_select.done_penalty=2000.0" \
    "train.ppo.eval_select.min_score_improvement=0.02" \
    "train.ppo.eval_select.final_eval=True" \
    "train.ppo.eval_select.save_deploy_best=True"
}

algo_class() {
  case "$1" in
    padapt) echo "ProprioAdapt" ;;
    purebc) echo "PureBC" ;;
    diffusion_latent) echo "DiffusionLatentStudent" ;;
    consistency_latent) echo "ConsistencyLatentStudent" ;;
    flow_matching) echo "FlowMatchingLatentStudent" ;;
    diffusion_action_chunk_len1|diffusion_action_chunk) echo "DiffusionActionChunkStudent" ;;
    teacher_ppo) echo "PPO" ;;
    bc_latentbc) echo "BCStudent" ;;
    dagger) echo "DAggerStudent" ;;
    dotpg) echo "DOTPG" ;;
    *) echo "unknown method: $1" >&2; return 2 ;;
  esac
}

nn_subdir() {
  case "$1" in
    padapt) echo "stage2_nn" ;;
    purebc) echo "stage2_bc_nn" ;;
    diffusion_latent) echo "stage2_diffusion_nn" ;;
    consistency_latent) echo "stage2_consistency_nn" ;;
    flow_matching) echo "stage2_flow_nn" ;;
    diffusion_action_chunk_len1|diffusion_action_chunk) echo "stage2_diffusion_action_chunk_nn" ;;
    *) echo "" ;;
  esac
}

algo_extra() {
  case "$1" in
    padapt|purebc)
      true
      ;;
    diffusion_latent)
      printf '%s\n' \
        "+train.ppo.diffusion_steps=10" \
        "+train.ppo.diffusion_steps_infer=10" \
        "+train.ppo.diffusion_lr=3e-4" \
        "+train.ppo.diffusion_loss_coef=1.0" \
        "+train.ppo.bc_loss_coef=1.0"
      ;;
    consistency_latent)
      printf '%s\n' \
        "+train.ppo.consistency_lr=3e-4" \
        "+train.ppo.consistency_loss_coef=1.0" \
        "+train.ppo.consistency_boundary_coef=0.5" \
        "+train.ppo.consistency_num_scales=10" \
        "+train.ppo.consistency_infer_steps=1" \
        "+train.ppo.bc_loss_coef=1.0"
      ;;
    flow_matching)
      printf '%s\n' \
        "+train.ppo.flow_lr=3e-4" \
        "+train.ppo.flow_loss_coef=1.0" \
        "+train.ppo.flow_infer_steps=1" \
        "+train.ppo.bc_loss_coef=1.0"
      ;;
    diffusion_action_chunk_len1)
      printf '%s\n' \
        "+train.ppo.action_chunk_len=1" \
        "+train.ppo.action_chunk_diffusion_steps=10" \
        "+train.ppo.action_chunk_diffusion_steps_infer=10" \
        "+train.ppo.action_chunk_diffusion_lr=3e-4" \
        "+train.ppo.action_chunk_diffusion_loss_coef=1.0" \
        "+train.ppo.action_chunk_first_action_bc_loss_coef=1.0" \
        "+train.ppo.action_chunk_bc_loss_coef=0.1" \
        "+train.ppo.action_chunk_stochastic_infer=False" \
        "+train.ppo.action_chunk_teacher_mix_steps=120000"
      ;;
    diffusion_action_chunk)
      printf '%s\n' \
        "+train.ppo.action_chunk_len=8" \
        "+train.ppo.action_chunk_diffusion_steps=10" \
        "+train.ppo.action_chunk_diffusion_steps_infer=10" \
        "+train.ppo.action_chunk_diffusion_lr=3e-4" \
        "+train.ppo.action_chunk_diffusion_loss_coef=1.0" \
        "+train.ppo.action_chunk_first_action_bc_loss_coef=1.0" \
        "+train.ppo.action_chunk_bc_loss_coef=0.1" \
        "+train.ppo.action_chunk_stochastic_infer=False" \
        "+train.ppo.action_chunk_teacher_mix_steps=120000"
      ;;
    dotpg)
      printf '%s\n' \
        "++train.dotpg.state_mode=student" \
        "++train.dotpg.dynamic_state=True" \
        "++train.dotpg.policy_arch=teacher_actor" \
        "++train.dotpg.policy_init_from_teacher=True" \
        "++train.dotpg.policy_output_mode=clamp" \
        "++train.dotpg.policy_loss_mode=dual" \
        "++train.dotpg.policy_dual_coef=1.0" \
        "++train.dotpg.dual_state_scale=1.0" \
        "++train.dotpg.dual_action_scale=1.0" \
        "++train.dotpg.critic_state_scale=1.0" \
        "++train.dotpg.critic_action_scale=1.0"
      ;;
    *) true ;;
  esac
}

checkpoint_for_trained() {
  local method="$1"
  local seed="$2"
  local output="outputs/${RUN_NAME}/train_outputs/${method}_s${seed}"
  local nn
  nn="$(nn_subdir "${method}")"
  echo "${output}/${nn}/model_best_deploy.ckpt"
}

run_train_job() {
  local method="$1"
  local seed="$2"
  local output="${RUN_NAME}/train_outputs/${method}_s${seed}"
  local log="${RUN_DIR}/logs/train_${method}_s${seed}.log"
  local status_file="${RUN_DIR}/status/train_${method}_s${seed}.status"
  local class
  class="$(algo_class "${method}")"
  mapfile -t common < <(common_overrides)
  mapfile -t evalsel < <(eval_select_overrides)
  mapfile -t extra < <(algo_extra "${method}")
  local cmd=(
    timeout "${WINDOW_SEC}"
    python train.py
    "${common[@]}"
    "seed=${seed}"
    "train.algo=${class}"
    "train.ppo.proprio_adapt=True"
    "train.ppo.output_name=${output}"
    "experiment=${RUN_NAME}_${method}_s${seed}"
    "checkpoint=${TEACHER_CKPT}"
    "${evalsel[@]}"
    "${extra[@]}"
  )
  {
    echo "[train] method=${method} seed=${seed} start=$(date -Iseconds)"
    echo "[train] output=outputs/${output}"
    printf '[train] command: '; qcmd "${cmd[@]}"
    set +e
    CUDA_VISIBLE_DEVICES="${GPU}" "${cmd[@]}"
    local status=$?
    set -e
    echo "${status}" > "${status_file}"
    echo "[train] status=${status} end=$(date -Iseconds)"
    find "outputs/${output}" -maxdepth 3 -type f \( -name '*.ckpt' -o -name 'config_*.yaml' -o -name 'eval_select_history.tsv' \) -printf '%p\t%s\n' 2>/dev/null | sort || true
    if [[ "${status}" -ne 0 && "${status}" -ne 124 ]]; then
      exit "${status}"
    fi
  } > "${log}" 2>&1
}

run_train_batches() {
  local jobs=("$@")
  local pids=()
  local running=0
  for item in "${jobs[@]}"; do
    local method="${item%%:*}"
    local seed="${item##*:}"
    run_train_job "${method}" "${seed}" &
    pids+=("$!")
    running=$((running + 1))
    if [[ "${running}" -ge "${PARALLEL_TRAIN}" ]]; then
      for pid in "${pids[@]}"; do wait "${pid}"; done
      pids=()
      running=0
    fi
  done
  for pid in "${pids[@]}"; do wait "${pid}"; done
}

run_eval_one() {
  local group="$1"
  local method="$2"
  local train_seed="$3"
  local eval_seed="$4"
  local ckpt="$5"
  shift 5
  local extra=("$@")
  local class
  class="$(algo_class "${method}")"
  local json="${RUN_DIR}/raw_json/${group}_${method}_train${train_seed}_eval${eval_seed}.json"
  local csv="${RUN_DIR}/raw_csv/${group}_append.csv"
  local log="${RUN_DIR}/logs/${group}_${method}_train${train_seed}_eval${eval_seed}.log"
  mapfile -t common < <(common_overrides)
  local student_restore=()
  if [[ "${method}" != "teacher_ppo" ]]; then
    student_restore=("train.ppo.proprio_adapt=True")
  fi
  local cmd=(
    python scripts/paper_codrive_eval.py
    --method "${method}"
    --train-seed "${train_seed}"
    --algo "${class}"
    --checkpoint "${ckpt}"
    --json-out "${json}"
    --csv-out "${csv}"
    --fixed-steps 2048
    --episode-target 256
    --episode-max-steps 8192
    "${common[@]}"
    "seed=${eval_seed}"
    "train.algo=${class}"
    "train.ppo.output_name=${RUN_NAME}/eval_tmp/${group}_${method}_train${train_seed}_eval${eval_seed}"
    "${student_restore[@]}"
    "${extra[@]}"
  )
  printf '%s\t%s\t%s\t%s\t%s\t' "$(date -Iseconds)" "${group}" "${method}" "${train_seed}" "${eval_seed}" >> "${COMMANDS_LOG}"
  qcmd "${cmd[@]}" >> "${COMMANDS_LOG}"
  {
    echo "[eval] group=${group} method=${method} train_seed=${train_seed} eval_seed=${eval_seed} start=$(date -Iseconds)"
    echo "[eval] ckpt=${ckpt}"
    printf '[eval] command: '; qcmd "${cmd[@]}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${cmd[@]}"
    echo "[eval] end=$(date -Iseconds)"
  } > "${log}" 2>&1
}

run_latency_one() {
  local group="$1"
  local method="$2"
  local train_seed="$3"
  local batch="$4"
  local ckpt="$5"
  shift 5
  local extra=("$@")
  local class
  class="$(algo_class "${method}")"
  local json="${RUN_DIR}/raw_json/${group}_${method}_train${train_seed}_batch${batch}.json"
  local log="${RUN_DIR}/logs/${group}_${method}_train${train_seed}_batch${batch}.log"
  mapfile -t common < <(common_overrides)
  local student_restore=()
  if [[ "${method}" != "teacher_ppo" ]]; then
    student_restore=("train.ppo.proprio_adapt=True")
  fi
  local cmd=(
    python scripts/paper_codrive_eval.py
    --mode latency
    --method "${method}"
    --train-seed "${train_seed}"
    --algo "${class}"
    --checkpoint "${ckpt}"
    --json-out "${json}"
    --latency-warmup 200
    --latency-measure 1000
    "${common[@]}"
    "seed=42"
    "task.env.numEnvs=${batch}"
    "train.algo=${class}"
    "train.ppo.output_name=${RUN_NAME}/eval_tmp/${group}_${method}_train${train_seed}_batch${batch}"
    "${student_restore[@]}"
    "${extra[@]}"
  )
  printf '%s\t%s\t%s\t%s\t%s\t' "$(date -Iseconds)" "${group}" "${method}" "${train_seed}" "${batch}" >> "${COMMANDS_LOG}"
  qcmd "${cmd[@]}" >> "${COMMANDS_LOG}"
  {
    echo "[latency] group=${group} method=${method} train_seed=${train_seed} batch=${batch} start=$(date -Iseconds)"
    printf '[latency] command: '; qcmd "${cmd[@]}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${cmd[@]}"
    echo "[latency] end=$(date -Iseconds)"
  } > "${log}" 2>&1
}

summarize_group() {
  local group="$1"
  python scripts/paper_codrive_summarize.py \
    --json-glob "${RUN_DIR}/raw_json/${group}_*.json" \
    --raw-out "${RUN_DIR}/raw_csv/${group}_raw.csv" \
    --agg-out "${RUN_DIR}/aggregate_csv/${group}_aggregate.csv" \
    --group-by method mode
}

validate_logs() {
  local out="${RUN_DIR}/validation_summary.txt"
  {
    echo "time=$(date -Iseconds)"
    echo "run_dir=${RUN_DIR}"
    echo
    echo "[bad_patterns]"
    grep -RInE 'Traceback|RuntimeError|CUDA out of memory|Missing key\\(s\\)|Unexpected key\\(s\\)|FileNotFoundError|Error executing job' "${RUN_DIR}/logs" || true
    echo
    echo "[missing_deploy_ckpts]"
    for method in padapt purebc diffusion_latent consistency_latent flow_matching; do
      for seed in 42 43 44; do
        ckpt="$(checkpoint_for_trained "${method}" "${seed}")"
        [[ -f "${ckpt}" ]] || echo "${method} ${seed} ${ckpt}"
      done
    done
  } > "${out}"
}

append_handoff_start() {
  {
    echo
    echo "## $(date +%F) -- CoDriveThesis Full Paper Experiment Started"
    echo
    echo "- Run dir: \`${RUN_DIR}\`"
    echo "- tmux/session owner: \`${SESSION_NAME:-manual}\`"
    echo "- Plan: formal 5-method 3-train-seed suite, unified eval, NFE/latency, representation ablation, robustness."
    echo "- GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
    echo "- Status file: \`${RUN_DIR}/status/phase.txt\`"
  } >> docs/cloud_session_handoff.md
}

main() {
  write_manifest
  append_handoff_start
  monitor_gpu > "${RUN_DIR}/gpu_monitor.tsv" 2>&1 &
  local monitor_pid=$!
  trap 'kill "${monitor_pid}" 2>/dev/null || true' EXIT

  log_phase "preflight"
  if pgrep -af 'python train.py' >/dev/null; then
    echo "Existing train.py process found; aborting." | tee "${RUN_DIR}/status/preflight_error.txt"
    exit 3
  fi
  [[ -f "${TEACHER_CKPT}" ]] || { echo "missing teacher ${TEACHER_CKPT}"; exit 2; }
  finish_phase "preflight" "ok"

  log_phase "formal_training"
  local train_jobs=()
  for method in padapt purebc diffusion_latent consistency_latent flow_matching; do
    for seed in 42 43 44; do
      train_jobs+=("${method}:${seed}")
    done
  done
  run_train_batches "${train_jobs[@]}"
  finish_phase "formal_training" "done"

  log_phase "representation_train"
  run_train_batches "diffusion_action_chunk_len1:42"
  finish_phase "representation_train" "done"

  log_phase "main_eval"
  for method in padapt purebc diffusion_latent consistency_latent flow_matching; do
    mapfile -t extra < <(algo_extra "${method}")
    for train_seed in 42 43 44; do
      ckpt="$(checkpoint_for_trained "${method}" "${train_seed}")"
      for eval_seed in 42 43 44; do
        run_eval_one "main" "${method}" "${train_seed}" "${eval_seed}" "${ckpt}" "${extra[@]}"
      done
    done
  done
  run_eval_one "main" "teacher_ppo" "teacher3655" "42" "${TEACHER_CKPT}"
  run_eval_one "main" "teacher_ppo" "teacher3655" "43" "${TEACHER_CKPT}"
  run_eval_one "main" "teacher_ppo" "teacher3655" "44" "${TEACHER_CKPT}"
  run_eval_one "main" "bc_latentbc" "42" "42" "outputs/Dexh13HoraLightbulb_student_bc_codrive_thesis/codrive_thesis_formal_latentbc_s42/bc_nn/model_best_student_eval.ckpt" "train.ppo.proprio_adapt=True"
  run_eval_one "main" "bc_latentbc" "42" "43" "outputs/Dexh13HoraLightbulb_student_bc_codrive_thesis/codrive_thesis_formal_latentbc_s42/bc_nn/model_best_student_eval.ckpt" "train.ppo.proprio_adapt=True"
  run_eval_one "main" "bc_latentbc" "42" "44" "outputs/Dexh13HoraLightbulb_student_bc_codrive_thesis/codrive_thesis_formal_latentbc_s42/bc_nn/model_best_student_eval.ckpt" "train.ppo.proprio_adapt=True"
  run_eval_one "main" "dagger" "42" "42" "outputs/Dexh13HoraLightbulb_student_dagger_codrive_thesis/codrive_thesis_formal_dagger_pure_replay_s42/dagger_nn/model_best_student_eval.ckpt" "train.ppo.proprio_adapt=True"
  run_eval_one "main" "dagger" "42" "43" "outputs/Dexh13HoraLightbulb_student_dagger_codrive_thesis/codrive_thesis_formal_dagger_pure_replay_s42/dagger_nn/model_best_student_eval.ckpt" "train.ppo.proprio_adapt=True"
  run_eval_one "main" "dagger" "42" "44" "outputs/Dexh13HoraLightbulb_student_dagger_codrive_thesis/codrive_thesis_formal_dagger_pure_replay_s42/dagger_nn/model_best_student_eval.ckpt" "train.ppo.proprio_adapt=True"
  mapfile -t dotpg_extra < <(algo_extra "dotpg")
  run_eval_one "main" "dotpg" "42" "42" "outputs/Dexh13HoraLightbulb_student_dotpg_codrive_thesis/codrive_thesis_formal_dotpg_dual_bc5_s42/student_output/dotpg_nn/model_best.ckpt" "${dotpg_extra[@]}"
  run_eval_one "main" "dotpg" "42" "43" "outputs/Dexh13HoraLightbulb_student_dotpg_codrive_thesis/codrive_thesis_formal_dotpg_dual_bc5_s42/student_output/dotpg_nn/model_best.ckpt" "${dotpg_extra[@]}"
  run_eval_one "main" "dotpg" "42" "44" "outputs/Dexh13HoraLightbulb_student_dotpg_codrive_thesis/codrive_thesis_formal_dotpg_dual_bc5_s42/student_output/dotpg_nn/model_best.ckpt" "${dotpg_extra[@]}"
  summarize_group "main"
  finish_phase "main_eval" "done"

  log_phase "nfe_latency"
  local nfe
  for method in diffusion_latent consistency_latent flow_matching diffusion_action_chunk; do
    local ckpt
    if [[ "${method}" == "diffusion_action_chunk" ]]; then
      ckpt="outputs/Dexh13HoraLightbulb_student_diffusion_action_chunk_codrive_thesis/codrive_thesis_teacher3655_s42_3h_final_20260504/stage2_diffusion_action_chunk_nn/model_best_train.ckpt"
    else
      ckpt="$(checkpoint_for_trained "${method}" "42")"
    fi
    for nfe in 1 2 4 8 10; do
      local nfe_extra=()
      case "${method}" in
        diffusion_latent) nfe_extra=("+train.ppo.diffusion_steps=10" "+train.ppo.diffusion_steps_infer=${nfe}" "+train.ppo.diffusion_lr=3e-4" "+train.ppo.diffusion_loss_coef=1.0" "+train.ppo.bc_loss_coef=1.0") ;;
        consistency_latent) nfe_extra=("+train.ppo.consistency_lr=3e-4" "+train.ppo.consistency_loss_coef=1.0" "+train.ppo.consistency_boundary_coef=0.5" "+train.ppo.consistency_num_scales=10" "+train.ppo.consistency_infer_steps=${nfe}" "+train.ppo.bc_loss_coef=1.0") ;;
        flow_matching) nfe_extra=("+train.ppo.flow_lr=3e-4" "+train.ppo.flow_loss_coef=1.0" "+train.ppo.flow_infer_steps=${nfe}" "+train.ppo.bc_loss_coef=1.0") ;;
        diffusion_action_chunk) nfe_extra=("+train.ppo.action_chunk_len=8" "+train.ppo.action_chunk_diffusion_steps=10" "+train.ppo.action_chunk_diffusion_steps_infer=${nfe}" "+train.ppo.action_chunk_diffusion_lr=3e-4" "+train.ppo.action_chunk_diffusion_loss_coef=1.0" "+train.ppo.action_chunk_first_action_bc_loss_coef=1.0" "+train.ppo.action_chunk_bc_loss_coef=0.1" "+train.ppo.action_chunk_stochastic_infer=False" "+train.ppo.action_chunk_teacher_mix_steps=120000") ;;
      esac
      for eval_seed in 42 43 44; do
        run_eval_one "nfe${nfe}" "${method}" "42" "${eval_seed}" "${ckpt}" "${nfe_extra[@]}"
      done
      for batch in 1 48 256; do
        run_latency_one "latency_nfe${nfe}" "${method}" "42" "${batch}" "${ckpt}" "${nfe_extra[@]}"
      done
    done
  done
  summarize_group "nfe1"; summarize_group "nfe2"; summarize_group "nfe4"; summarize_group "nfe8"; summarize_group "nfe10"
  summarize_group "latency_nfe1"; summarize_group "latency_nfe2"; summarize_group "latency_nfe4"; summarize_group "latency_nfe8"; summarize_group "latency_nfe10"
  finish_phase "nfe_latency" "done"

  log_phase "representation_eval"
  ckpt="$(checkpoint_for_trained "diffusion_action_chunk_len1" "42")"
  mapfile -t ac1_extra < <(algo_extra "diffusion_action_chunk_len1")
  for eval_seed in 42 43 44; do
    run_eval_one "representation" "diffusion_action_chunk_len1" "42" "${eval_seed}" "${ckpt}" "${ac1_extra[@]}"
  done
  mapfile -t ac8_extra < <(algo_extra "diffusion_action_chunk")
  for eval_seed in 42 43 44; do
    run_eval_one "representation" "diffusion_action_chunk" "42" "${eval_seed}" "outputs/Dexh13HoraLightbulb_student_diffusion_action_chunk_codrive_thesis/codrive_thesis_teacher3655_s42_3h_final_20260504/stage2_diffusion_action_chunk_nn/model_best_train.ckpt" "${ac8_extra[@]}"
  done
  summarize_group "representation"
  finish_phase "representation_eval" "done"

  log_phase "robustness"
  local robust_methods=(teacher_ppo padapt purebc diffusion_latent consistency_latent flow_matching)
  local stress_names=(nominal obs2x obs4x friction_wide masscom_wide initpos_noise)
  for method in "${robust_methods[@]}"; do
    local ckpt train_seed
    train_seed="42"
    case "${method}" in
      teacher_ppo) ckpt="${TEACHER_CKPT}"; train_seed="teacher3655" ;;
      *) ckpt="$(checkpoint_for_trained "${method}" "42")" ;;
    esac
    mapfile -t base_extra < <(algo_extra "${method}")
    for stress in "${stress_names[@]}"; do
      local stress_extra=()
      case "${stress}" in
        nominal) stress_extra=() ;;
        obs2x) stress_extra=("task.env.randomization.obs_noise_t_scale=0.02" "task.env.randomization.obs_noise_e_scale=0.04") ;;
        obs4x) stress_extra=("task.env.randomization.obs_noise_t_scale=0.04" "task.env.randomization.obs_noise_e_scale=0.08") ;;
        friction_wide) stress_extra=("task.env.randomization.randomizeFrictionLower=0.5" "task.env.randomization.randomizeFrictionUpper=7.0") ;;
        masscom_wide) stress_extra=("task.env.randomization.randomizeMassLower=0.03" "task.env.randomization.randomizeMassUpper=0.08" "task.env.randomization.randomizeCOMLower=-0.002" "task.env.randomization.randomizeCOMUpper=0.002") ;;
        initpos_noise) stress_extra=("task.env.object.init_pos_noise=[0.002,0.002,0.0]") ;;
      esac
      for eval_seed in 42 43 44; do
        run_eval_one "robust_${stress}" "${method}" "${train_seed}" "${eval_seed}" "${ckpt}" "${base_extra[@]}" "${stress_extra[@]}"
      done
    done
  done
  for stress in "${stress_names[@]}"; do summarize_group "robust_${stress}"; done
  finish_phase "robustness" "done"

  log_phase "validation"
  validate_logs
  printf 'done\n' > "${PHASE_FILE}"
  finish_phase "validation" "done"
  {
    echo
    echo "## $(date +%F) -- CoDriveThesis Full Paper Experiment Pipeline Finished"
    echo
    echo "- Run dir: \`${RUN_DIR}\`"
    echo "- Validation: \`${RUN_DIR}/validation_summary.txt\`"
    echo "- Main aggregate: \`${RUN_DIR}/aggregate_csv/main_aggregate.csv\`"
  } >> docs/cloud_session_handoff.md
}

main "$@"
