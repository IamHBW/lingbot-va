#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
rgbdwam_root="${RGBDWAM_ROOT:-$(cd "${repo_root}/../rgbdwam" && pwd)}"
server_script="${rgbdwam_root}/scripts/robotwin_distributed_server.sh"
topology="${ROBOTWIN_EVAL_TOPOLOGY:-htrain-local}"
phase="${ROBOTWIN_EVAL_PHASE:-clean}"
checkpoint="${CHECKPOINT:-}"
run_config="${RUN_CONFIG:-}"
robotwin_root="${ROBOTWIN_ROOT:-}"
lingbot_python="${LINGBOT_PYTHON:-/mnt/data/public_tools/miniconda3/envs/lingbot-va/bin/python}"
robotwin_python="${ROBOTWIN_PYTHON:-/mnt/data/public_tools/miniconda3/envs/RoboTwin/bin/python}"
server_base_port="${SERVER_BASE_PORT:-19000}"
episodes="${EPISODES_PER_TASK:-20}"
max_attempts="${MAX_TASK_ATTEMPTS:-3}"
run_id="${RUN_ID:-lingbot-${phase}-$(date -u +%Y%m%dT%H%M%SZ)}"
output_root="${OUTPUT_ROOT:-/mnt/data/users/${USER}/workspace/outputs/lingbot-va/robotwin-eval/${run_id}}"
export TMPDIR="${TMPDIR:-/mnt/data/users/${USER}/workspace/tmp/lingbot-va-eval/${run_id}}"

fail() {
    echo "[ERROR] $*" >&2
    exit 1
}

[[ "${topology}" == "htrain-local" || "${topology}" == "5090-remote" ]] \
    || fail "ROBOTWIN_EVAL_TOPOLOGY must be htrain-local or 5090-remote."
[[ -x "${server_script}" ]] || fail "Server script is not executable: ${server_script}"

if [[ "${topology}" == "5090-remote" ]]; then
    [[ "${NETWORK_MODE:-}" == "direct" || "${NETWORK_MODE:-}" == "relay" ]] \
        || fail "5090-remote requires NETWORK_MODE=direct or relay."
    exec env BACKEND=lingbot LINGBOT_ROOT="${repo_root}" LINGBOT_PYTHON="${lingbot_python}" \
        CHECKPOINT="${checkpoint}" RUN_CONFIG="${run_config}" ROBOTWIN_ROOT="${robotwin_root}" \
        bash "${server_script}"
fi

[[ "${phase}" == "clean" || "${phase}" == "randomized" ]] || fail "ROBOTWIN_EVAL_PHASE must be clean or randomized."
[[ "${episodes}" == "20" ]] || fail "Full evaluation requires EPISODES_PER_TASK=20."
[[ "${max_attempts}" =~ ^[1-9][0-9]*$ ]] || fail "MAX_TASK_ATTEMPTS must be positive."
[[ -d "${checkpoint}" ]] || fail "CHECKPOINT model root not found: ${checkpoint}"
[[ -f "${run_config}" ]] || fail "RUN_CONFIG not found: ${run_config}"
[[ -d "${robotwin_root}" ]] || fail "ROBOTWIN_ROOT not found: ${robotwin_root}"
[[ -x "${lingbot_python}" ]] || fail "LINGBOT_PYTHON is not executable: ${lingbot_python}"
[[ -x "${robotwin_python}" ]] || fail "ROBOTWIN_PYTHON is not executable: ${robotwin_python}"

mapfile -t tasks < <("${lingbot_python}" -c 'from evaluation.robotwin.eval_protocol import TASKS; print(*TASKS, sep="\n")')
[[ "${#tasks[@]}" == "50" ]] || fail "Expected exactly 50 tasks, got ${#tasks[@]}"
[[ "$(printf '%s\n' "${tasks[@]}" | sort -u | wc -l | tr -d ' ')" == "50" ]] || fail "Task list contains duplicates."

server_root="${output_root}/server"
server_pid=""
worker_pids=()
cleanup() {
    trap - EXIT INT TERM
    for pid in "${worker_pids[@]+"${worker_pids[@]}"}"; do
        kill "${pid}" 2>/dev/null || true
    done
    if [[ -n "${server_pid}" ]]; then
        kill "${server_pid}" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

mkdir -p "${output_root}/logs" "${output_root}/tasks" "${output_root}/sample_videos"
mkdir -p "${TMPDIR}"

env BACKEND=lingbot NETWORK_MODE=local LINGBOT_ROOT="${repo_root}" \
    LINGBOT_PYTHON="${lingbot_python}" CHECKPOINT="${checkpoint}" RUN_CONFIG="${run_config}" \
    ROBOTWIN_ROOT="${robotwin_root}" RGBDWAM_ROOT="${rgbdwam_root}" RUN_ID="${run_id}" \
    OUTPUT_ROOT="${server_root}" SERVER_BASE_PORT="${server_base_port}" SERVER_COUNT=8 \
    DRY_RUN="${DRY_RUN:-0}" bash "${server_script}" >"${output_root}/logs/server-supervisor.log" 2>&1 &
server_pid="$!"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    wait "${server_pid}"
    echo "[INFO] htrain-local dry run: phase=${phase} tasks=50 episodes_per_task=20 slots=8 output=${output_root}"
    exit 0
fi

for ((attempt = 0; attempt < 1800; ++attempt)); do
    [[ -f "${server_root}/server_manifest.json" ]] && break
    kill -0 "${server_pid}" 2>/dev/null || fail "Policy server supervisor exited during startup; see ${output_root}/logs/server-supervisor.log"
    sleep 2
done
[[ -f "${server_root}/server_manifest.json" ]] || fail "Policy servers did not become ready."
jq --arg phase "${phase}" --arg output_root "${output_root}" --arg tmpdir "${TMPDIR}" \
    '.evaluation.active_phase = $phase | .runtime = {output_root: $output_root, tmpdir: $tmpdir}' \
    "${server_root}/server_manifest.json" >"${output_root}/manifest.json"
echo "[INFO] eight LingBot servers ready; launching 50 ${phase} tasks"

task_config="demo_${phase}"
run_slot() {
    local slot="$1" index task task_root attempt_root attempt rc
    for ((index = slot; index < ${#tasks[@]}; index += 8)); do
        task="${tasks[$index]}"
        task_root="${output_root}/tasks/${task}"
        mkdir -p "${task_root}"
        for ((attempt = 1; attempt <= max_attempts; ++attempt)); do
            attempt_root="${task_root}/attempt-${attempt}"
            mkdir -p "${attempt_root}"
            echo "[INFO] slot=${slot} task=${task} phase=${phase} attempt=${attempt}" | tee -a "${output_root}/logs/slot${slot}.log"
            set +e
            (
                cd "${robotwin_root}"
                export CUDA_VISIBLE_DEVICES="${slot}"
                export PYTHONPATH="${repo_root}:${robotwin_root}:${PYTHONPATH:-}"
                export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
                export MPLCONFIGDIR="${output_root}/.matplotlib-slot${slot}"
                export ROBOTWIN_ROOT="${robotwin_root}"
                "${robotwin_python}" -u -m evaluation.robotwin.eval_polict_client_openpi \
                    --task-name "${task}" --task-config "${task_config}" \
                    --policy-name ACT --ckpt-setting step-30000 \
                    --host 127.0.0.1 --port "$((server_base_port + slot))" \
                    --save_root "${attempt_root}" --test_num "${episodes}" \
                    --instruction-type unseen --start-seed 100000 --phase "${phase}" \
                    --results-detailed "${attempt_root}/results_detailed.jsonl" \
                    --exclusions "${attempt_root}/exclusions.jsonl" \
                    --save-sample-videos --sample-video-root "${output_root}/sample_videos"
            ) >"${attempt_root}/client.log" 2>&1
            rc=$?
            set -e
            if [[ "${rc}" == "0" ]] && "${lingbot_python}" -m evaluation.robotwin.eval_protocol audit-attempt \
                --path "${attempt_root}/results_detailed.jsonl" --task "${task}" --phase "${phase}"; then
                printf '%s\n' "${attempt}" >"${task_root}/completed_attempt.txt"
                break
            fi
            printf 'task=%s phase=%s attempt=%s return_code=%s log=%s\n' \
                "${task}" "${phase}" "${attempt}" "${rc}" "${attempt_root}/client.log" \
                >>"${output_root}/failed_attempts.txt"
        done
        [[ -f "${task_root}/completed_attempt.txt" ]] || return 1
    done
}

for slot in {0..7}; do
    run_slot "${slot}" &
    worker_pids+=("$!")
done

workers_ok=1
for pid in "${worker_pids[@]}"; do
    wait "${pid}" || workers_ok=0
done
[[ "${workers_ok}" == "1" ]] || fail "At least one task failed all attempts; see ${output_root}/failed_attempts.txt"

"${lingbot_python}" -m evaluation.robotwin.eval_protocol aggregate-phase --root "${output_root}" --phase "${phase}"
touch "${server_root}/LOCAL_DONE"
wait "${server_pid}"
server_pid=""
trap - EXIT INT TERM
echo "[INFO] completed ${phase}: 1000 unique finished rollouts at ${output_root}"
