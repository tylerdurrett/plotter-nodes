# Implementation Guide Format

## Header

    # Implementation Guide: [Feature Title]

    **Date:** YYYY-MM-DD
    **Feature:** [Feature Name]
    **Source:** [relative link to source feature description or PRD]

## Overview

1-3 paragraphs describing the implementation approach, key sequencing
goals, and any architectural assumptions locked for this version.

## File Structure (when applicable)

ASCII diagram showing the planned file/directory layout. Include only
when the feature introduces new files or reorganizes existing ones.

## Phases

Organize implementation into sequential numbered phases. Each phase
contains numbered sub-sections.

### Phase Structure

    ## Phase N: [Phase Title]

    **Purpose:** One sentence — what this phase accomplishes.

    **Rationale:** Why this phase exists and why it is sequenced here.

    ### N.1 [Sub-section Title]

    - [ ] Task description (one story point)
    - [ ] Another task
    - [ ] Write unit/integration tests for [component]

    **Acceptance Criteria:**
    - Verifiable condition 1
    - Verifiable condition 2

### Phase Guidelines

- Each sub-section should represent roughly one story point of work.
- Number sub-sections within their phase: 1.1, 1.2, 2.1, 2.2, etc.
- Include test-writing tasks within each sub-section, not in a
  separate testing phase.
- Order tasks so each sub-section is independently testable upon
  completion.
- If database migrations are needed, batch related schema changes and
  place them in early phases.

## End Sections

### Dependency Graph

ASCII diagram showing phase and sub-section dependencies.

    ## Dependency Graph

    ```
    Phase 1 (Bootstrap)
      1.1 → 1.2 → 1.3
             |
        Phase 2
      2.1 → 2.2
    ```

### Key Design Decisions

Table summarizing important architectural choices and their rationale.

    ## Key Design Decisions

    | Decision | Rationale |
    |----------|-----------|
    | Choice 1 | Why this was chosen |
    | Choice 2 | Why this was chosen |

## Quality Checklist

- Phases are ordered for incremental testability
- Every sub-section has acceptance criteria
- Test creation is embedded in each phase, not deferred
- Database changes (if any) are batched and placed early
- Rationale is included for phase sequencing decisions
- Tasks are one story point or smaller
- Dependency graph reflects actual sequencing constraints
