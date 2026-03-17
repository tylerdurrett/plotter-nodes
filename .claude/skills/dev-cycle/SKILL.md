---
name: dev-cycle
description: Orchestrate the full feature development lifecycle from initial scoping through implementation to completion. Use when the user wants to start a new feature from scratch, resume work on an existing feature, check the status of features in the pipeline, or be guided through the next step in a feature's development process. Also use when the user invokes /dev-cycle.
---

# Dev Cycle

Orchestrate a feature through the full development lifecycle by
detecting its current phase and invoking the appropriate skill or
command at each step.

## Process

### 1. Identify the Feature

Determine which feature to work on:

- If the user names a feature or provides a path, use that.
- If ambiguous, scan `_tasks/` subdirectories to list active
  features and ask the user to pick one.
- If starting fresh (no existing feature), proceed to Scope phase.

### 2. Detect Current Phase

Locate the feature folder and determine its state:

| Folder location    | Files present                           | Current phase |
| ------------------ | --------------------------------------- | ------------- |
| Not yet created    | --                                      | Scope         |
| `_planning/`       | No implementation guide                 | Plan          |
| `_planning/`       | Has implementation guide                | Ready         |
| `_ready-to-start/` | --                                      | Start         |
| `_in-progress/`    | Unchecked tasks remain                  | Implement     |
| `_in-progress/`    | All tasks checked, docs not yet updated | Update Docs   |
| `_in-progress/`    | All tasks checked, docs updated         | Complete      |
| `_complete/`       | --                                      | Done          |

Report the detected phase to the user and confirm before proceeding.

### 3. Execute Current Phase

**Scope** (new feature):
Invoke `/feature-request`. When the feature description is written,
ask: "Feature description is done. Ready to create the implementation
plan?"

**Plan** (feature description exists, no implementation guide):
Invoke `/implementation-guide` using the feature description in the
feature folder. When complete, ask: "Implementation guide is done.
Ready to mark this feature as ready to start?"

**Ready** (implementation guide exists, still in `_planning/`):
Invoke `/set-status-ready` for the feature folder. Commit the
changes. Ask: "Feature is marked ready. Want to start implementation
now?"

**Start** (feature is in `_ready-to-start/`):
Invoke `/set-status-in-progress` for the feature folder. Commit the
move. Then proceed to the Implement phase.

**Implement** (feature is in `_in-progress/`):
Before starting, check runtime sync state via `/sync-status`. If
`blocked-dirty` or `blocked-diverged`, pause and report the block
with remediation (e.g., "commit or stash changes", "rebase or
reset"). Use `/sync-now` after resolving. Proceed on `in-sync`,
`syncing`, or `degraded-network`.
Invoke `/do-section` with the implementation guide. After each
section completes, check if more sections remain. If yes, ask:
"Section done. Continue with the next section?" If all sections are
complete, proceed to the Update Docs phase.

**Update Docs** (all implementation sections done, before completing):
Review what changed during implementation and update any affected
documentation (README, architecture diagrams, inline doc comments,
config references, etc.). Check `docs/` and the repo root for
existing docs that may need updates. Commit the doc changes. Then
ask: "Docs are updated. Ready to mark this feature as complete?"

**Complete** (all implementation tasks done, docs updated):
Invoke `/set-status-complete` for the feature folder. Commit the
move. Report: "Feature is complete."

### 4. Side Exits

At any transition point, the user may choose to:

- **Pause**: Stop here and resume later (no action needed).
- **Icebox**: Invoke `/set-status-icebox` to shelve the feature.
- **Abandon**: Invoke `/set-status-abandoned` to drop the feature.

Respect these choices without pushback.

### 5. Session Boundaries

Each phase may occur in a separate conversation. When resuming:

- Re-detect the feature folder location and current phase (step 2).
- Do not assume context from prior sessions.
- Orient the user with a brief status summary before continuing.
