#!/usr/bin/env bash
#
# remote_run.sh -- enforces the local-dev / remote-long-run workflow:
#   1. Refuses to run if the local working tree has uncommitted changes, or if local
#      HEAD isn't exactly what's on origin/main (catches "forgot to push").
#   2. git pull --ff-only on the remote lab GPU machine (hard-fails on any divergence
#      rather than silently merging -- the remote checkout should never be edited directly).
#   3. Launches the requested command in a detached tmux session there, so it survives
#      the SSH connection dropping and can be attached to live (tqdm progress etc.),
#      not just tailed after the fact.
#
# Usage:
#   scripts/remote_run.sh run <job_name> -- <command...>   Sync + launch a job
#   scripts/remote_run.sh list                              List running/finished jobs
#   scripts/remote_run.sh log <job_name> [N]                Tail last N lines (default 50)
#   scripts/remote_run.sh follow <job_name>                  Live-follow output (tail -f)
#   scripts/remote_run.sh attach <job_name>                  Interactively attach to the
#                                                             live tmux session (foreground;
#                                                             Ctrl-B D to detach without
#                                                             killing the job)
#   scripts/remote_run.sh kill <job_name>                    Kill a running job
#
# Examples:
#   scripts/remote_run.sh run formation_train -- \
#       scripts/torchrl/mappo_train.py --task Formation-TorchRL-UAVSwarm-Direct-v0 \
#       --experiment_directory formation_scalability --headless
#
#   scripts/remote_run.sh run formation_sweep -- \
#       scripts/torchrl/train_multi_seed.py --task Formation-TorchRL-UAVSwarm-Direct-v0 \
#       --experiment_directory formation_scalability --num_seeds 5 --headless

set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-lab-gpu}"
REMOTE_REPO="${REMOTE_REPO:-~/UavSwarm-baseline_branch}"
REMOTE_PY="${REMOTE_PY:-~/miniconda3/envs/isaac_env/bin/python}"
REMOTE_LOG_DIR="${REMOTE_REPO}/logs/remote_runs"
SESSION_PREFIX="uav_"

usage() {
    sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

# cd to repo root so this works regardless of invocation cwd.
cd "$(git rev-parse --show-toplevel)"

check_local_sync() {
    echo "[remote_run] Checking local repo state..."
    if [[ -n "$(git status --porcelain)" ]]; then
        echo "[remote_run] ERROR: local working tree has uncommitted changes. Commit first." >&2
        git status --short >&2
        exit 1
    fi
    git fetch origin main --quiet
    local local_head remote_head
    local_head=$(git rev-parse HEAD)
    remote_head=$(git rev-parse origin/main)
    if [[ "$local_head" != "$remote_head" ]]; then
        echo "[remote_run] ERROR: local HEAD ($local_head) != origin/main ($remote_head)." >&2
        echo "  Push local commits first, or pull if origin is ahead of local." >&2
        exit 1
    fi
    echo "[remote_run] Local is in sync with origin/main ($local_head)."
}

sync_remote() {
    echo "[remote_run] Pulling latest on $REMOTE_HOST..."
    ssh "$REMOTE_HOST" "cd $REMOTE_REPO && git fetch origin main --quiet && git pull --ff-only origin main"
    local remote_head
    remote_head=$(ssh "$REMOTE_HOST" "cd $REMOTE_REPO && git rev-parse HEAD")
    echo "[remote_run] Remote now at $remote_head"
}

cmd_run() {
    local job_name="$1"; shift
    [[ "${1:-}" == "--" ]] || usage
    shift
    [[ $# -ge 1 ]] || usage
    local session="${SESSION_PREFIX}${job_name}"

    if ssh "$REMOTE_HOST" "tmux has-session -t '${session}' 2>/dev/null"; then
        echo "[remote_run] ERROR: job '${job_name}' is already running on ${REMOTE_HOST}." >&2
        echo "  Use a different job name, or 'kill ${job_name}' first." >&2
        exit 1
    fi

    check_local_sync
    sync_remote

    local timestamp remote_log runner_script
    timestamp=$(date +%Y%m%d_%H%M%S)
    remote_log="${REMOTE_LOG_DIR}/${job_name}_${timestamp}.log"
    runner_script="${REMOTE_REPO}/.remote_jobs/${job_name}_run.sh"

    # Build the command as a *file* on the remote rather than one big quoted string --
    # embedding "$@" via $* (or any single-string reconstruction) loses per-argument
    # quoting, and tmux's `new-session -d -s NAME "STRING"` re-parses STRING through a
    # nested shell anyway, so any shell metacharacter inside an argument (quotes,
    # semicolons, brackets -- e.g. in a `-c "..."` snippet) breaks silently. printf %q
    # safely escapes each argument for exactly one shell re-parse (the script itself),
    # and tmux only ever has to run a plain `bash <path>` -- no nested quoting at all.
    local py_cmd
    py_cmd=$(printf '%q ' "$REMOTE_PY" -u "$@")
    local script_content
    script_content=$(cat <<SCRIPT
#!/usr/bin/env bash
cd $(printf '%q' "$REMOTE_REPO")
${py_cmd}2>&1 | tee $(printf '%q' "$remote_log")
SCRIPT
)
    ssh "$REMOTE_HOST" "mkdir -p '${REMOTE_LOG_DIR}' '${REMOTE_REPO}/.remote_jobs'"
    printf '%s\n' "$script_content" | ssh "$REMOTE_HOST" "cat > '${runner_script}' && chmod +x '${runner_script}'"
    ssh "$REMOTE_HOST" "tmux new-session -d -s '${session}' 'bash ${runner_script}'"
    echo "[remote_run] Launched '${job_name}' in tmux session '${session}' on ${REMOTE_HOST}."
    echo "  Log:            ${remote_log}"
    echo "  Check progress: scripts/remote_run.sh log ${job_name}"
    echo "  Live-follow:    scripts/remote_run.sh follow ${job_name}"
    echo "  True attach:    scripts/remote_run.sh attach ${job_name}"
}

cmd_list() {
    echo "[remote_run] tmux sessions on ${REMOTE_HOST}:"
    ssh "$REMOTE_HOST" "tmux list-sessions 2>/dev/null | grep '^${SESSION_PREFIX}' || echo '  (none running)'"
}

latest_log() {
    # $1 = job_name. Prints the path of the most recent log file for that job.
    ssh "$REMOTE_HOST" "ls -t '${REMOTE_LOG_DIR}/$1_'*.log 2>/dev/null | head -1"
}

cmd_log() {
    local job_name="$1"
    local n="${2:-50}"
    local latest
    latest=$(latest_log "$job_name")
    if [[ -z "$latest" ]]; then
        echo "[remote_run] No log found for job '${job_name}'." >&2
        exit 1
    fi
    echo "[remote_run] Tailing ${latest} (last ${n} lines):"
    ssh "$REMOTE_HOST" "tail -n '${n}' '${latest}'"
}

cmd_follow() {
    local job_name="$1"
    local latest
    latest=$(latest_log "$job_name")
    if [[ -z "$latest" ]]; then
        echo "[remote_run] No log found for job '${job_name}'." >&2
        exit 1
    fi
    echo "[remote_run] Following ${latest} (Ctrl-C to stop watching -- job keeps running)..."
    ssh "$REMOTE_HOST" "tail -n 50 -f '${latest}'"
}

cmd_attach() {
    local job_name="$1"
    local session="${SESSION_PREFIX}${job_name}"
    echo "[remote_run] Attaching to '${session}' (Ctrl-B D to detach without killing it)..."
    ssh -t "$REMOTE_HOST" "tmux attach -t '${session}'"
}

cmd_kill() {
    local job_name="$1"
    local session="${SESSION_PREFIX}${job_name}"
    if ssh "$REMOTE_HOST" "tmux kill-session -t '${session}' 2>/dev/null"; then
        echo "[remote_run] Killed session '${session}'."
    else
        echo "[remote_run] No session named '${session}' found." >&2
        exit 1
    fi
}

[[ $# -ge 1 ]] || usage
subcmd="$1"; shift
case "$subcmd" in
    run)    [[ $# -ge 1 ]] || usage; job="$1"; shift; cmd_run "$job" "$@" ;;
    list)   cmd_list ;;
    log)    [[ $# -ge 1 ]] || usage; cmd_log "$@" ;;
    follow) [[ $# -ge 1 ]] || usage; cmd_follow "$1" ;;
    attach) [[ $# -ge 1 ]] || usage; cmd_attach "$1" ;;
    kill)   [[ $# -ge 1 ]] || usage; cmd_kill "$1" ;;
    *) usage ;;
esac
