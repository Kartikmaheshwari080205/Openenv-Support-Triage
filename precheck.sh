#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-openenv-support-triage}"
CONTAINER_NAME="${2:-openenv-support-triage-precheck}"
PORT="${3:-7860}"
MODEL="${4:-gpt-4o-mini}"

assert_condition() {
  local condition="$1"
  local message="$2"
  if [[ "$condition" != "true" ]]; then
    echo "$message" >&2
    exit 1
  fi
}

if command -v openenv >/dev/null 2>&1; then
  echo "[0/5] Running openenv validate..."
  openenv validate
else
  echo "[0/5] openenv CLI not found; skipping openenv validate"
fi

echo "[1/5] Building Docker image..."
docker build -t "$IMAGE_NAME" .

echo "[2/5] Starting container..."
docker run --rm -d -p "${PORT}:7860" --name "$CONTAINER_NAME" "$IMAGE_NAME" >/dev/null

cleanup() {
  echo "Stopping container..."
  docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

sleep 2

echo "[3/5] Running endpoint checks..."

root_status="$(curl -sS "http://localhost:${PORT}/" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("status",""))')"
assert_condition "$([[ "$root_status" == "ok" ]] && echo true || echo false)" "Root endpoint check failed"

reset_json='{"task_id":"support-easy-001"}'
reset_task_id="$(curl -sS -X POST "http://localhost:${PORT}/reset" -H 'Content-Type: application/json' -d "$reset_json" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("task_id",""))')"
assert_condition "$([[ "$reset_task_id" == "support-easy-001" ]] && echo true || echo false)" "Reset endpoint returned unexpected task_id"

tasks_count="$(curl -sS "http://localhost:${PORT}/tasks" | python3 -c 'import json,sys; print(len(json.load(sys.stdin).get("tasks",[])))')"
assert_condition "$([[ "$tasks_count" -ge 3 ]] && echo true || echo false)" "Tasks endpoint returned fewer than 3 tasks"

grader_score="$(curl -sS "http://localhost:${PORT}/grader" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("score",-1))')"
assert_condition "$(python3 - <<PY
score = float("$grader_score")
print("true" if 0.0 <= score <= 1.0 else "false")
PY
)" "Grader score out of [0,1]"

baseline_payload="{\"model\":\"${MODEL}\"}"
baseline_response="$(curl -sS -X POST "http://localhost:${PORT}/baseline" -H 'Content-Type: application/json' -d "$baseline_payload")"

baseline_task_count="$(printf '%s' "$baseline_response" | python3 -c 'import json,sys; print(len(json.load(sys.stdin).get("task_results",[])))')"
assert_condition "$([[ "$baseline_task_count" -ge 3 ]] && echo true || echo false)" "Baseline result missing task scores"

baseline_avg="$(printf '%s' "$baseline_response" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("average_score",-1))')"
assert_condition "$(python3 - <<PY
score = float("$baseline_avg")
print("true" if 0.0 <= score <= 1.0 else "false")
PY
)" "Baseline average score out of [0,1]"

echo "[5/5] Precheck passed"
printf '%s\n' "$baseline_response" | python3 -m json.tool
