#!/usr/bin/env bash
# ralph_json.sh - Iterative Claude Code runner for JSON-based task queues
# Two-phase loop: plan (read-only) -> execute (auto-accept)
# Streams Claude's output in real-time so you can watch progress.
#
# Usage:
#   ./ralph_json.sh <json-file> <prompt-file> [max-iterations] [extra-instructions] [--stop-on-manual-test] [--no-plan]
#
# Arguments:
#   json-file:             Path to a JSON file containing an array of task items.
#   prompt-file:           Path to a file containing the instruction prompt for the agent.
#   max-iterations:        Maximum iterations to run (default: 30). Each iteration processes one item.
#   extra-instructions:    Additional instructions appended to the prompt (optional, quote the string).
#   --stop-on-manual-test: Stop when NEEDS_HUMAN_TEST is emitted (default: off).
#   --no-plan:             Skip the planning phase, run everything in a single execution pass.
#
# JSON File Format:
#   The JSON file must be an array of objects. Each object MUST have a "status" field.
#   Valid statuses: "unprocessed", "complete", "error", "needs_human_test"
#   Only items with status "unprocessed" will be picked up for processing.
#
#   Example:
#     [
#       {"title": "Task A", "_id": "task-a", "status": "unprocessed"},
#       {"title": "Task B", "_id": "task-b", "status": "complete", "output": {"key": "value"}},
#       {"title": "Task C", "_id": "task-c", "status": "error", "error": "Something failed"}
#     ]
#
# Agent Output Markers:
#   The agent communicates results using these conventions:
#
#   Stop codes (exactly one required per item):
#     <promise>COMPLETE</promise>     - Item task completed successfully.
#     <promise>HALT</promise>         - Item encountered an unresolvable error.
#     <promise>NEEDS_HUMAN_TEST</promise> - Item requires manual testing.
#
#   Data tags (optional, captured and saved to the item):
#     <output>...</output>  - Output data (JSON or text) saved to the item's "output" field.
#     <error>...</error>    - Error description saved to the item's "error" field (used with HALT).
#
# Exit Codes:
#   0 - All unprocessed items have been handled (or none to process).
#   1 - Max iterations reached with unprocessed items remaining.
#   3 - NEEDS_HUMAN_TEST triggered with --stop-on-manual-test enabled.

set -uo pipefail
# Note: not using set -e — we handle errors explicitly to avoid silent exits.

# --- Argument parsing ---

if [ -z "${1:-}" ] || [ -z "${2:-}" ]; then
  echo "Usage: $0 <json-file> <prompt-file> [max-iterations] [extra-instructions] [--stop-on-manual-test] [--no-plan]"
  echo "  json-file:           Path to a JSON array of task items (each must have a 'status' field)"
  echo "  prompt-file:         Path to a file containing the instruction prompt"
  echo "  max-iterations:      Maximum iterations to run (default: 30)"
  echo "  extra-instructions:  Additional instructions appended to the prompt (optional, quote the string)"
  echo "  --stop-on-manual-test: Stop when NEEDS_HUMAN_TEST is emitted (default: off)"
  echo "  --no-plan:           Skip the planning phase, run in a single execution pass"
  exit 1
fi

JSON_FILE="$1"
shift
PROMPT_FILE="$1"
shift

MAX_ITERATIONS="30"
EXTRA_INSTRUCTIONS=""
STOP_ON_MANUAL_TEST=0
NO_PLAN=0

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
    --no-plan)
      NO_PLAN=1
      ;;
    *)
      echo "Error: Unknown argument: $1"
      echo "Usage: $0 <json-file> <prompt-file> [max-iterations] [extra-instructions] [--stop-on-manual-test] [--no-plan]"
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

# --- Constants ---

COMPLETE_MARKER='<promise>COMPLETE</promise>'
HALT_MARKER='<promise>HALT</promise>'
NEEDS_HUMAN_TEST_MARKER='<promise>NEEDS_HUMAN_TEST</promise>'
ALLOWED_TOOLS="Edit,Write,Bash,Read,Glob,Grep,WebFetch,WebSearch,TodoWrite,Task,NotebookEdit"
TMPFILE=$(mktemp)
TMPJSON=$(mktemp)
trap 'rm -f "$TMPFILE" "$TMPJSON"' EXIT

# --- Validation ---

if [ ! -f "$JSON_FILE" ]; then
  echo "Error: JSON file not found: $JSON_FILE"
  exit 1
fi

if [ ! -f "$PROMPT_FILE" ]; then
  echo "Error: Prompt file not found: $PROMPT_FILE"
  exit 1
fi

# Validate JSON structure
if ! jq empty "$JSON_FILE" 2>/dev/null; then
  echo "Error: Invalid JSON in: $JSON_FILE"
  exit 1
fi

json_type=$(jq -r 'type' "$JSON_FILE")
if [ "$json_type" != "array" ]; then
  echo "Error: JSON must be an array, got: $json_type"
  exit 1
fi

missing_status=$(jq '[to_entries[] | select(.value.status == null) | .key] | length' "$JSON_FILE")
if [ "$missing_status" -gt 0 ]; then
  echo "Error: $missing_status item(s) missing the required 'status' field at indices:"
  jq '[to_entries[] | select(.value.status == null) | .key]' "$JSON_FILE"
  exit 1
fi

echo "JSON validation passed."

# --- Read instruction prompt ---

INSTRUCTION_PROMPT=$(cat "$PROMPT_FILE")

# --- Build manual test prompt line ---

if [ "$STOP_ON_MANUAL_TEST" -eq 1 ]; then
  MANUAL_TEST_PROMPT_LINE="- $NEEDS_HUMAN_TEST_MARKER -- Output when you have completed implementation but it requires manual human testing that cannot be automated (e.g., UI verification, hardware interaction, external service integration). Include a description of exactly what needs to be tested alongside the marker."
else
  MANUAL_TEST_PROMPT_LINE="- $NEEDS_HUMAN_TEST_MARKER -- Do NOT output this marker. If the item needs manual human testing, record the test steps and continue. Mark it as $COMPLETE_MARKER instead."
fi

# --- Helper functions ---

# Run claude and stream output in real-time.
# Captures raw stream-json to tmpfile while displaying assistant text live.
run_claude_streaming() {
  local tmpfile="$1"
  shift
  > "$tmpfile"

  "$CLAUDE" "$@" --output-format stream-json --verbose \
    | tee "$tmpfile" \
    | jq --unbuffered -r 'select(.type == "assistant") | .message.content[]? | select(.type == "text") | .text' 2>/dev/null \
    || true

  if [ ! -s "$tmpfile" ]; then
    echo "[ralph_json] Warning: No output from claude. Retrying with json format for diagnostics..."
    "$CLAUDE" "$@" --output-format json 2>&1 | tee "$tmpfile" || true
  fi
}

# Extract session_id from stream-json (or json) output
get_session_id() {
  local sid
  sid=$(jq -r 'select(.session_id != null) | .session_id' "$1" 2>/dev/null | tail -1) || true
  if [ -z "$sid" ] || [ "$sid" = "null" ]; then
    sid=$(jq -r '.session_id // empty' "$1" 2>/dev/null) || true
  fi
  echo "${sid:-}"
}

# Extract final result text from stream-json (or json) output
get_result_text() {
  local txt
  txt=$(jq -r 'select(.type == "result") | .result' "$1" 2>/dev/null | tail -1) || true
  if [ -z "$txt" ]; then
    txt=$(jq -r '.result // empty' "$1" 2>/dev/null) || true
  fi
  echo "${txt:-}"
}

# Extract content between XML-style tags (supports multi-line).
# Usage: extract_tag "output" "$result_text"
extract_tag() {
  local tag="$1" text="$2"
  echo "$text" | perl -0777 -ne "print \$1 if /<${tag}>(.*?)<\/${tag}>/s"
}

# Get the index of the next unprocessed item. Returns empty string if none.
get_next_unprocessed_index() {
  local idx
  idx=$(jq '[to_entries[] | select(.value.status == "unprocessed") | .key] | .[0] // empty' "$JSON_FILE") || true
  echo "${idx:-}"
}

# Get the JSON of the item at a given index.
get_item_at_index() {
  jq ".[$1]" "$JSON_FILE"
}

# Update an item's status in the JSON file.
# Usage: update_item_status <index> <status>
update_item_status() {
  local index="$1" status="$2"
  jq --arg s "$status" ".[$index].status = \$s" "$JSON_FILE" > "$TMPJSON" && mv "$TMPJSON" "$JSON_FILE"
}

# Set an item's error field.
# Usage: set_item_error <index> <error_message>
set_item_error() {
  local index="$1" error_msg="$2"
  jq --arg e "$error_msg" ".[$index].error = \$e" "$JSON_FILE" > "$TMPJSON" && mv "$TMPJSON" "$JSON_FILE"
}

# Set an item's output field. Attempts to parse as JSON; falls back to string.
# Usage: set_item_output <index> <output_text>
set_item_output() {
  local index="$1" output_text="$2"
  # Try to parse as JSON first
  if echo "$output_text" | jq empty 2>/dev/null; then
    jq --argjson o "$output_text" ".[$index].output = \$o" "$JSON_FILE" > "$TMPJSON" && mv "$TMPJSON" "$JSON_FILE"
  else
    jq --arg o "$output_text" ".[$index].output = \$o" "$JSON_FILE" > "$TMPJSON" && mv "$TMPJSON" "$JSON_FILE"
  fi
}

# Check result text for stop codes and update the item accordingly.
# Returns: 0 if a marker was found and handled, 1 if no marker found.
# Sets global ITEM_HANDLED=1 when a marker is found.
handle_item_result() {
  local result="$1"
  local index="$2"
  local phase="$3"
  local iteration="$4"

  # Extract output tag if present (applies to any stop code)
  local output_content
  output_content=$(extract_tag "output" "$result")

  if [[ "$result" == *"$HALT_MARKER"* ]]; then
    echo ""
    echo "[ralph_json] Item $index: HALT detected ($phase, iteration $iteration)."
    update_item_status "$index" "error"
    local error_content
    error_content=$(extract_tag "error" "$result")
    if [ -n "$error_content" ]; then
      set_item_error "$index" "$error_content"
      echo "[ralph_json] Error saved: $error_content"
    else
      set_item_error "$index" "Agent halted without providing an error description"
    fi
    if [ -n "$output_content" ]; then
      set_item_output "$index" "$output_content"
    fi
    ITEM_HANDLED=1
    return 0
  fi

  if [[ "$result" == *"$NEEDS_HUMAN_TEST_MARKER"* ]]; then
    echo ""
    echo "[ralph_json] Item $index: NEEDS_HUMAN_TEST detected ($phase, iteration $iteration)."
    update_item_status "$index" "needs_human_test"
    if [ -n "$output_content" ]; then
      set_item_output "$index" "$output_content"
    fi
    ITEM_HANDLED=1
    if [ "$STOP_ON_MANUAL_TEST" -eq 1 ]; then
      echo "[ralph_json] --stop-on-manual-test is enabled. Stopping."
      echo "$result"
      echo "=== Re-run ralph_json.sh to continue after testing. ==="
      exit 3
    fi
    echo "[ralph_json] Continuing to next item (stop-on-manual-test is disabled)."
    return 0
  fi

  if [[ "$result" == *"$COMPLETE_MARKER"* ]]; then
    echo ""
    echo "[ralph_json] Item $index: COMPLETE ($phase, iteration $iteration)."
    update_item_status "$index" "complete"
    if [ -n "$output_content" ]; then
      set_item_output "$index" "$output_content"
      echo "[ralph_json] Output saved."
    fi
    ITEM_HANDLED=1
    return 0
  fi

  # No marker found
  return 1
}

# --- Print configuration ---

echo "JSON file: $JSON_FILE"
echo "Prompt file: $PROMPT_FILE"
echo "Max iterations: $MAX_ITERATIONS"
if [ "$STOP_ON_MANUAL_TEST" -eq 1 ]; then
  echo "Stop on manual test: enabled"
else
  echo "Stop on manual test: disabled (default)"
fi
if [ "$NO_PLAN" -eq 1 ]; then
  echo "Plan phase: skipped (--no-plan)"
fi
if [ -n "$EXTRA_INSTRUCTIONS" ]; then
  echo "Extra instructions: $EXTRA_INSTRUCTIONS"
fi

total_items=$(jq 'length' "$JSON_FILE")
unprocessed_count=$(jq '[.[] | select(.status == "unprocessed")] | length' "$JSON_FILE")
echo "Total items: $total_items ($unprocessed_count unprocessed)"
echo "---"

# --- Main loop ---

for ((i=1; i<=MAX_ITERATIONS; i++)); do
  # Find next unprocessed item
  item_index=$(get_next_unprocessed_index)

  if [ -z "$item_index" ]; then
    remaining=$(jq '[.[] | select(.status == "unprocessed")] | length' "$JSON_FILE")
    complete=$(jq '[.[] | select(.status == "complete")] | length' "$JSON_FILE")
    errors=$(jq '[.[] | select(.status == "error")] | length' "$JSON_FILE")
    needs_test=$(jq '[.[] | select(.status == "needs_human_test")] | length' "$JSON_FILE")
    echo ""
    echo "=== No more unprocessed items. ==="
    echo "Summary: $complete complete, $errors errors, $needs_test needs_human_test, $remaining unprocessed"
    exit 0
  fi

  item_json=$(get_item_at_index "$item_index")
  item_title=$(echo "$item_json" | jq -r '.title // .name // ._id // "unknown"')

  echo ""
  echo "=== Iteration $i / $MAX_ITERATIONS — Item $item_index: $item_title ==="

  # Build prompt for this item
  ITEM_PROMPT=$(cat <<EOF
Here is your task item:
$item_json

You will work on that item (and only that item).

$INSTRUCTION_PROMPT

IMPORTANT: STOP CODES: You MUST output exactly one of these stop codes when done:

- $COMPLETE_MARKER -- Output when you have successfully completed the task for this item.
- $HALT_MARKER -- Output when you encounter an error or blocker you cannot resolve. Wrap the error description in <error>...</error> tags.
$MANUAL_TEST_PROMPT_LINE

IMPORTANT: OUTPUT: If your task produces output data, wrap it in <output>...</output> tags (e.g., <output>{"key": "value"}</output>). The script will capture this and save it to the item.
${EXTRA_INSTRUCTIONS:+
ADDITIONAL INSTRUCTIONS: $EXTRA_INSTRUCTIONS}
EOF
)

  ITEM_HANDLED=0

  if [ "$NO_PLAN" -eq 0 ]; then
    # Phase 1: Plan (read-only, streamed)
    echo "--- Planning phase ---"
    run_claude_streaming "$TMPFILE" -p "$ITEM_PROMPT" --permission-mode plan

    session_id=$(get_session_id "$TMPFILE")
    plan_result=$(get_result_text "$TMPFILE")

    if [ -z "$plan_result" ]; then
      echo "[ralph_json] Warning: Could not extract plan result. Raw output (first 20 lines):"
      head -20 "$TMPFILE"
    fi

    # Check if planning phase produced a stop code
    if [ -n "$plan_result" ]; then
      handle_item_result "$plan_result" "$item_index" "planning phase" "$i" && continue
    fi

    if [ -z "$session_id" ] || [ "$session_id" = "null" ]; then
      echo "[ralph_json] Error: Failed to get session ID from planning phase."
      echo "[ralph_json] Marking item $item_index as error."
      update_item_status "$item_index" "error"
      set_item_error "$item_index" "Failed to get session ID from planning phase"
      continue
    fi
  fi

  # Phase 2: Execute (auto-accept tools, resume plan session, streamed)
  echo ""
  if [ "$NO_PLAN" -eq 0 ]; then
    echo "--- Execution phase (session: $session_id) ---"
    run_claude_streaming "$TMPFILE" \
      -p "Execute the plan you just created. Implement the changes, then provide your stop code and any output." \
      --resume "$session_id" \
      --allowedTools "$ALLOWED_TOOLS"
  else
    echo "--- Execution phase (no plan) ---"
    run_claude_streaming "$TMPFILE" \
      -p "$ITEM_PROMPT" \
      --allowedTools "$ALLOWED_TOOLS"
    session_id=$(get_session_id "$TMPFILE")
  fi

  exec_result=$(get_result_text "$TMPFILE")

  # Check if execution phase produced a stop code
  if [ -n "$exec_result" ]; then
    handle_item_result "$exec_result" "$item_index" "execution phase" "$i" && continue
  fi

  # Phase 3: Retry — no marker found, ask the agent explicitly
  echo ""
  echo "[ralph_json] No stop code detected. Asking agent for status..."
  run_claude_streaming "$TMPFILE" \
    -p "You did not provide a stop code. Please provide exactly one of these markers to indicate your status: $COMPLETE_MARKER (if the task is done), $HALT_MARKER (if there was an error — include <error>description</error>), or $NEEDS_HUMAN_TEST_MARKER (if manual testing is needed). Also include <output>...</output> if you have output data." \
    --resume "$session_id" \
    --allowedTools "$ALLOWED_TOOLS"

  retry_result=$(get_result_text "$TMPFILE")

  if [ -n "$retry_result" ]; then
    handle_item_result "$retry_result" "$item_index" "retry phase" "$i" && continue
  fi

  # Still no marker — mark as error
  echo "[ralph_json] Still no stop code after retry. Marking item $item_index as error."
  update_item_status "$item_index" "error"
  set_item_error "$item_index" "No completion marker received after retry"
done

# Print final summary
remaining=$(jq '[.[] | select(.status == "unprocessed")] | length' "$JSON_FILE")
complete=$(jq '[.[] | select(.status == "complete")] | length' "$JSON_FILE")
errors=$(jq '[.[] | select(.status == "error")] | length' "$JSON_FILE")
needs_test=$(jq '[.[] | select(.status == "needs_human_test")] | length' "$JSON_FILE")
echo ""
echo "=== Reached max iterations ($MAX_ITERATIONS) with $remaining unprocessed items remaining. ==="
echo "Summary: $complete complete, $errors errors, $needs_test needs_human_test, $remaining unprocessed"
exit 1
