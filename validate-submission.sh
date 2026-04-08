#!/usr/bin/env bash
# validate-submission.sh — Validate the algo_reasoning_env submission
#
# Usage:
#   ./validate-submission.sh [SERVER_URL]
#
# Checks:
#   1. Server is reachable at the given URL (default: http://localhost:7860)
#   2. POST /reset returns a valid observation with session_id
#   3. POST /step returns a valid graded response
#   4. openenv.yaml exists and has correct port

PASS=0
FAIL=0

green() { echo -e "\033[32mPASS\033[0m $1"; PASS=$((PASS + 1)); }
red()   { echo -e "\033[31mFAIL\033[0m $1"; FAIL=$((FAIL + 1)); }

echo "========================================"
echo "  Algo Reasoning Env — Submission Check"
echo "========================================"
echo ""

SERVER_URL="${1:-http://localhost:7860}"

# --- Check 1: Server reachable ---
echo "[1/4] Server reachable at ${SERVER_URL}"
HEALTH=$(curl -sf --max-time 5 "${SERVER_URL}/health" 2>/dev/null && echo "ok" || echo "fail")
if [ "$HEALTH" = "ok" ]; then
    green "Server is reachable"
else
    # Try /reset as fallback
    RESET_CHECK=$(curl -sf --max-time 10 -X POST "${SERVER_URL}/reset" \
        -H "Content-Type: application/json" -d '{}' 2>/dev/null || echo "")
    if [ -n "$RESET_CHECK" ]; then
        green "Server is reachable (via /reset)"
    else
        red "Server not reachable at ${SERVER_URL}"
        echo "  Start the server with:"
        echo "    uvicorn algo_reasoning_env.server.app:app --host 0.0.0.0 --port 7860"
        echo ""
        echo "Results: ${PASS}/4 passed, ${FAIL}/4 failed"
        exit 1
    fi
fi

# --- Check 2: POST /reset returns observation ---
echo "[2/4] POST /reset returns valid observation"
RESET_RESP=$(curl -sf --max-time 10 -X POST "${SERVER_URL}/reset" \
    -H "Content-Type: application/json" -d '{}' 2>/dev/null || echo "")

if echo "$RESET_RESP" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    obs = data.get('observation', data)
    has_session = bool(data.get('session_id'))
    has_problem_id = bool(obs.get('problem_id'))
    has_diff = bool(obs.get('difficulty'))
    has_desc = bool(obs.get('problem_description'))
    ok = has_session and (has_problem_id or has_diff or has_desc)
    sys.exit(0 if ok else 1)
except Exception as e:
    sys.exit(1)
" 2>/dev/null; then
    green "POST /reset returns valid observation"
else
    red "POST /reset did not return valid observation"
    echo "  Response: $(echo "$RESET_RESP" | head -c 300)"
fi

# --- Check 3: POST /step returns graded response ---
echo "[3/4] POST /step returns graded response"

# Reset to get a fresh session_id
RESET_FOR_STEP=$(curl -sf --max-time 10 -X POST "${SERVER_URL}/reset" \
    -H "Content-Type: application/json" -d '{}' 2>/dev/null || echo "")

SESSION_ID=$(echo "$RESET_FOR_STEP" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('session_id', ''))
except:
    pass
" 2>/dev/null)

if [ -z "$SESSION_ID" ]; then
    red "POST /step did not return graded response (no session_id from reset)"
else
    STEP_RESP=$(curl -sf --max-time 60 -X POST "${SERVER_URL}/step" \
        -H "Content-Type: application/json" \
        -d "{\"session_id\": \"${SESSION_ID}\", \"action\": {\"solution_code\": \"impl Solution { pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> { vec![] } }\", \"reasoning_steps\": \"step-1: Create result vector\", \"time_complexity\": \"O(n)\"}}" 2>/dev/null || echo "")

    if echo "$STEP_RESP" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    obs = data.get('observation', data)
    has_reward = data.get('reward') is not None
    has_correctness = 'correctness_reward' in obs
    has_reasoning = 'reasoning_score' in obs
    has_grading = has_reward or has_correctness or has_reasoning
    sys.exit(0 if has_grading else 1)
except Exception as e:
    sys.exit(1)
" 2>/dev/null; then
        green "POST /step returns graded response"
    else
        red "POST /step did not return graded response"
        echo "  Response: $(echo "$STEP_RESP" | head -c 300)"
    fi
fi

# --- Check 4: openenv.yaml has port 7860 ---
echo "[4/4] openenv.yaml has correct port"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
YAML_FILE="${SCRIPT_DIR}/openenv.yaml"

if [ -f "$YAML_FILE" ]; then
    if grep -q "port: 7860" "$YAML_FILE"; then
        green "openenv.yaml has port 7860"
    else
        red "openenv.yaml does not have port 7860"
        echo "  Contents:"
        cat "$YAML_FILE"
    fi
else
    red "openenv.yaml not found at ${YAML_FILE}"
fi

# --- Summary ---
echo ""
echo "========================================"
echo "  Results: ${PASS}/4 passed, ${FAIL}/4 failed"
echo "========================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
