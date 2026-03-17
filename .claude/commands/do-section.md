The user wants you to implement one part of an implementation plan doc. They are providing the doc. If they haven't provided one, ask for it first.

1. Please read the plan carefully to fully understand the objective.
2. Find the next unfinished phase section.
3. Check runtime sync state before proceeding. If sync is blocked, stop and report:
   - Run `/sync-status` (or check the sync gate output if visible).
   - If status is `blocked-dirty` or `blocked-diverged`, do NOT proceed. Report the block to the user with the remediation steps (e.g., "commit or stash changes", "rebase or reset"). Suggest `/sync-now` after the issue is resolved.
   - If status is `in-sync`, `syncing`, or `degraded-network`, proceed normally.
4. Read through the codebase as needed to fully understand how the system works and the existing codebase conventions. If a tweak to the plan is needed based on the codebase conventions, please do so.

Our goal is to implement the plan using standard best practices and clean, maintainable, well-documented code following the established codebase patterns. Please implement this next unfinished section of the phase. ONLY ONE SECTION.

As much as possible, perform your own testing and acceptance criteria validation, unless manual user testing is required.

ALWAYS run the code quality review after you finish implementation.

When you're done, please update the implementation doc by checking off any completed tasks and adding notes where anything might have diverged from the original plan.
