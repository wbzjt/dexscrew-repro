#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?usage: paper_codrive_checkpoint_aliaser.sh RUN_DIR}"
INTERVAL_SEC="${INTERVAL_SEC:-60}"
STATUS_DIR="${RUN_DIR}/status"
ALIAS_LOG="${STATUS_DIR}/checkpoint_aliases.tsv"
HOTFIX_NOTE="${STATUS_DIR}/checkpoint_selection_hotfix.txt"

mkdir -p "${STATUS_DIR}"

if [[ ! -f "${HOTFIX_NOTE}" ]]; then
  {
    echo "time=$(date -Iseconds)"
    echo "reason=The formal training jobs are externally bounded by timeout, and the active run used eval_select.interval_agent_steps=20000000, while 5h runs reach roughly 4M agent steps. The planned model_best_deploy.ckpt is therefore not emitted before timeout."
    echo "action=Create a relative symlink model_best_deploy.ckpt -> model_best_train.ckpt for each completed/active formal run that has model_best_train.ckpt but no deploy checkpoint."
    echo "scope=${RUN_DIR}/train_outputs/*/*/model_best_train.ckpt"
    echo "paper_note=This makes checkpoint selection train-best for the active run. The deviation must be reported in the experiment manifest/results notes."
  } > "${HOTFIX_NOTE}"
fi

if [[ ! -f "${ALIAS_LOG}" ]]; then
  printf 'time\ttrain_ckpt\tdeploy_ckpt\taction\n' > "${ALIAS_LOG}"
fi

while true; do
  while IFS= read -r -d '' train_ckpt; do
    dir="$(dirname "${train_ckpt}")"
    deploy_ckpt="${dir}/model_best_deploy.ckpt"
    if [[ ! -e "${deploy_ckpt}" && ! -L "${deploy_ckpt}" ]]; then
      (
        cd "${dir}"
        ln -s "model_best_train.ckpt" "model_best_deploy.ckpt"
      )
      printf '%s\t%s\t%s\tcreated_symlink\n' "$(date -Iseconds)" "${train_ckpt}" "${deploy_ckpt}" >> "${ALIAS_LOG}"
    fi
  done < <(find "${RUN_DIR}/train_outputs" -type f -name 'model_best_train.ckpt' -print0 2>/dev/null || true)

  phase="$(cat "${RUN_DIR}/status/phase.txt" 2>/dev/null || true)"
  if [[ "${phase}" == "done" ]]; then
    exit 0
  fi
  sleep "${INTERVAL_SEC}"
done
