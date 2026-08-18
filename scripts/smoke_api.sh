#!/usr/bin/env bash
# Stage 2 smoke test. Assumes `make serve` is already running.
#   bash scripts/smoke_api.sh [port]
set -uo pipefail
PORT="${1:-8080}"
BASE="http://127.0.0.1:${PORT}"
PY="$(dirname "$0")/../.venv/bin/python"
fails=0

check() { # name expected_code url [curl args...]
  local name="$1" want="$2" url="$3"; shift 3
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' "$@" "$url")
  if [[ "$code" == "$want" ]]; then
    printf '  \033[32m✓\033[0m %-34s %s\n' "$name" "$code"
  else
    printf '  \033[31m✗\033[0m %-34s got %s, want %s\n' "$name" "$code" "$want"; fails=$((fails+1))
  fi
}

echo "Smoke testing ${BASE}"
check "GET  /health"            200 "$BASE/health"
check "GET  /  (dashboard)"     200 "$BASE/"
check "GET  /runs"              200 "$BASE/runs"
check "POST /infer"             200 "$BASE/infer" -X POST -H 'Content-Type: application/json' \
      -d '{"instruction":"move above the red block"}'
check "POST /infer (bad image)" 422 "$BASE/infer" -X POST -H 'Content-Type: application/json' \
      -d '{"instruction":"x","image_b64":"not-base64!!"}'
check "GET  /runs/<bogus>"      404 "$BASE/runs/does-not-exist"

# Traversal may be stopped either by the route regex ([^/]+ never matches an
# encoded slash -> 404) or by the explicit guard in get_frame -> 400. Both are
# correct; the only unacceptable answer is 200.
check_blocked() {
  local name="$1" url="$2"
  local code; code=$(curl -s -o /dev/null -w '%{http_code}' "$url")
  if [[ "$code" != "200" ]]; then
    printf '  \033[32m✓\033[0m %-34s blocked (%s)\n' "$name" "$code"
  else
    printf '  \033[31m✗\033[0m %-34s SERVED 200 — path traversal!\n' "$name"; fails=$((fails+1))
  fi
}
check "POST /episode"           200 "$BASE/episode" -X POST -H 'Content-Type: application/json' \
      -d '{"inject":"grasp_fail","max_steps":200}'

RID=$(curl -s "$BASE/runs?limit=1" | "$PY" -c 'import sys,json;d=json.load(sys.stdin);print(d[0]["run_id"] if d else "")')
if [[ -n "$RID" ]]; then
  check "GET  /runs/<id>"         200 "$BASE/runs/$RID"
  check "GET  /runs/<id>/gif"     200 "$BASE/runs/$RID/frames/rollout.gif"
  check "GET  /runs/<id>/actions" 200 "$BASE/runs/$RID/actions.jsonl"
  check_blocked "frame traversal"      "$BASE/runs/$RID/frames/..%2F..%2Fsummary.json"
  check_blocked "run-id traversal"     "$BASE/runs/..%2F..%2Fetc"
fi

echo
if (( fails )); then echo "  ✗ ${fails} 项失败"; exit 1; fi
echo "  ✓ 全部通过"
