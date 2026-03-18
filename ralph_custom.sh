#!/usr/bin/env bash
# ralph_custom.sh - Iterative Claude Code runner with a custom instruction prompt
# Two-phase loop: plan (read-only) -> execute (auto-accept)
# Streams Claude's output in real-time so you can watch progress.
# Usage: ./ralph_custom.sh <prompt-file> [max-iterations] [extra-instructions] [--stop-on-manual-test]

set -uo pipefail
# Note: not using set -e — we handle errors explicitly to avoid silent exits.

if [ -z "${1:-}" ]; then
  echo "Usage: $0 <prompt-file> [max-iterations] [extra-instructions] [--stop-on-manual-test]"
  echo "  prompt-file:         Path to a file containing the instruction prompt"
  echo "  max-iterations:      Maximum iterations to run (default: 30)"
  echo "  extra-instructions:  Additional instructions appended to the prompt (optional, quote the string)"
  echo "  --stop-on-manual-test: Stop when NEEDS_HUMAN_TEST is emitted (default: off)"
  exit 1
fi

PROMPT_FILE="$1"
shift

MAX_ITERATIONS="30"
EXTRA_INSTRUCTIONS=""
STOP_ON_MANUAL_TEST=0

if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
  MAX_ITERATIONS="$1"
  shift
fi

if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
  EXTRA_INSTRUCTIONS="$1"
  shift
fi

while [ $# -gt 0 ]; do
  case "$1" in
    --stop-on-manual-test)
      STOP_ON_MANUAL_TEST=1
      ;;
    *)
      echo "Error: Unknown argument: $1"
      echo "Usage: $0 <prompt-file> [max-iterations] [extra-instructions] [--stop-on-manual-test]"
      exit 1
      ;;
  esac
  shift
done

# --- Resolve claude binary ---
CLAUDE="${CLAUDE:-$(command -v claude 2>/dev/null || echo "$HOME/.claude/local/claude")}"
if [ ! -x "$CLAUDE" ]; then
  echo "Error: claude CLI not found. Set CLAUDE env var or ensure claude is in PATH."
  exit 1
fi

COMPLETE_MARKER='<promise>COMPLETE</promise>'
HALT_MARKER='<promise>HALT</promise>'
NEEDS_HUMAN_TEST_MARKER='<promise>NEEDS_HUMAN_TEST</promise>'
ALLOWED_TOOLS="Edit,Write,Bash,Read,Glob,Grep,WebFetch,WebSearch,TodoWrite,Task,NotebookEdit"
TMPFILE=$(mktemp)
trap 'rm -f "$TMPFILE"' EXIT

if [ ! -f "$PROMPT_FILE" ]; then
  echo "Error: Prompt file not found: $PROMPT_FILE"
  exit 1
fi

INSTRUCTION_PROMPT=$(cat "$PROMPT_FILE")

if [ "$STOP_ON_MANUAL_TEST" -eq 1 ]; then
  MANUAL_TEST_PROMPT_LINE="- $NEEDS_HUMAN_TEST_MARKER -- Output when you have completed implementation of a section but it requires manual human testing that cannot be automated (e.g., UI verification, hardware interaction, external service integration). Include a description of exactly what needs to be tested alongside the marker."
else
  MANUAL_TEST_PROMPT_LINE="- $NEEDS_HUMAN_TEST_MARKER -- Do NOT output this marker. If a section needs manual human testing, record exact manual test steps in the implementation guide notes and continue with the next unfinished section."
fi

PROMPT=$(cat <<EOF
$INSTRUCTION_PROMPT

IMPORTANT: STOP CODES: You MUST output one of these stop codes when the corresponding condition is met (but you should NOT output any of these codes until the condition is met):

- $COMPLETE_MARKER -- Output when ALL items in the entire implementation guide are complete.
- $HALT_MARKER -- Output when you encounter an error or blocker you cannot resolve on your own (e.g., missing dependencies, permissions issues, broken external services, fundamental misunderstanding in the plan). Include a clear description of the problem alongside the marker.
$MANUAL_TEST_PROMPT_LINE
${EXTRA_INSTRUCTIONS:+
ADDITIONAL INSTRUCTIONS: $EXTRA_INSTRUCTIONS}
EOF
)

# Run claude and stream output in real-time.
# Captures raw stream-json to tmpfile while displaying assistant text live.
run_claude_streaming() {
  local tmpfile="$1"
  shift
  > "$tmpfile"

  # Run claude with streaming. --verbose is required for stream-json.
  "$CLAUDE" "$@" --output-format stream-json --verbose \
    | tee "$tmpfile" \
    | jq --unbuffered -r 'select(.type == "assistant") | .message.content[]? | select(.type == "text") | .text' 2>/dev/null \
    || true

  # Debug: if tmpfile is empty, claude probably failed
  if [ ! -s "$tmpfile" ]; then
    echo "[ralph] Warning: No output from claude. Retrying with json format for diagnostics..."
    "$CLAUDE" "$@" --output-format json 2>&1 | tee "$tmpfile" || true
  fi
}

# Extract session_id from stream-json (or json) output
get_session_id() {
  # Try stream-json format first (multiple lines, each a JSON object)
  local sid
  sid=$(jq -r 'select(.session_id != null) | .session_id' "$1" 2>/dev/null | tail -1) || true
  # Fall back to single json object format
  if [ -z "$sid" ] || [ "$sid" = "null" ]; then
    sid=$(jq -r '.session_id // empty' "$1" 2>/dev/null) || true
  fi
  echo "${sid:-}"
}

# Extract final result text from stream-json (or json) output
get_result_text() {
  local txt
  # Try stream-json format (look for result event)
  txt=$(jq -r 'select(.type == "result") | .result' "$1" 2>/dev/null | tail -1) || true
  # Fall back to single json object format
  if [ -z "$txt" ]; then
    txt=$(jq -r '.result // empty' "$1" 2>/dev/null) || true
  fi
  echo "${txt:-}"
}

# Check result text for stop codes. Handles HALT, NEEDS_HUMAN_TEST, and COMPLETE.
# Args: $1 = result text, $2 = phase label (for log messages), $3 = iteration number
check_stop_codes() {
  local result="$1"
  local phase="$2"
  local iteration="$3"

  if [[ "$result" == *"$HALT_MARKER"* ]]; then
    echo ""
    echo "=== HALT: Unresolvable error detected ($phase, iteration $iteration). ==="
    echo "$result"
    exit 2
  fi

  if [[ "$result" == *"$NEEDS_HUMAN_TEST_MARKER"* ]]; then
    if [ "$STOP_ON_MANUAL_TEST" -eq 1 ]; then
      echo ""
      echo "=== Human testing required ($phase, iteration $iteration). ==="
      echo "$result"
      echo "=== Re-run ralph_custom.sh to continue after testing. ==="
      exit 3
    fi

    echo "[ralph] Notice: NEEDS_HUMAN_TEST marker detected in $phase (iteration $iteration), but stop-on-manual-test is disabled; continuing."
  fi

  if [[ "$result" == *"$COMPLETE_MARKER"* ]]; then
    echo ""
    echo "=== Task complete after $iteration iterations ($phase). ==="
    exit 0
  fi
}

echo "Prompt file: $PROMPT_FILE"
echo "Max iterations: $MAX_ITERATIONS"
if [ "$STOP_ON_MANUAL_TEST" -eq 1 ]; then
  echo "Stop on manual test: enabled"
else
  echo "Stop on manual test: disabled (default)"
fi
if [ -n "$EXTRA_INSTRUCTIONS" ]; then
  echo "Extra instructions: $EXTRA_INSTRUCTIONS"
fi
echo "---"

for ((i=1; i<=MAX_ITERATIONS; i++)); do
  echo ""
  echo "=== Iteration $i / $MAX_ITERATIONS ==="

  # Phase 1: Plan (read-only, streamed)
  echo "--- Planning phase ---"
  run_claude_streaming "$TMPFILE" -p "$PROMPT" --permission-mode plan

  session_id=$(get_session_id "$TMPFILE")
  plan_result=$(get_result_text "$TMPFILE")

  if [ -z "$plan_result" ]; then
    echo "[ralph] Warning: Could not extract plan result. Raw output (first 20 lines):"
    head -20 "$TMPFILE"
  fi

  check_stop_codes "${plan_result:-}" "planning phase" "$i"

  if [ -z "$session_id" ] || [ "$session_id" = "null" ]; then
    echo "[ralph] Error: Failed to get session ID from planning phase."
    echo "[ralph] Raw output (first 20 lines):"
    head -20 "$TMPFILE"
    exit 1
  fi

  # Phase 2: Execute (auto-accept tools, resume plan session, streamed)
  echo ""
  echo "--- Execution phase (session: $session_id) ---"
  run_claude_streaming "$TMPFILE" \
    -p "Execute the plan you just created. Implement the changes, then commit." \
    --resume "$session_id" \
    --allowedTools "$ALLOWED_TOOLS"

  exec_result=$(get_result_text "$TMPFILE")

  check_stop_codes "${exec_result:-}" "execution phase" "$i"
done

echo "=== Reached max iterations ($MAX_ITERATIONS) without completion. ==="
exit 1
