---
name: implementation-guide
description: Create detailed, phased implementation plans from feature descriptions or PRD documents. Use when the user wants to turn a scoped feature description into an actionable implementation guide with phased tasks, acceptance criteria, and testing strategy, or when they invoke the /implementation-guide command. Outputs an implementation guide markdown file colocated with the source feature description.
---

# Implementation Guide

Transform a scoped feature description or PRD into a detailed, phased
implementation plan with one-story-point tasks, acceptance criteria, and
built-in testing at every stage.

## Process

### 1. Gather Input

Obtain the feature description or PRD document. If the user has not
provided one, ask for it before proceeding.

### 2. Research

Ground in the codebase before planning. Understand:

- Existing architecture, conventions, and file structure
- Related functionality and integration points
- Testing patterns and frameworks in use
- Database/migration patterns (if applicable)

### 3. Plan Construction

Build the implementation guide following these principles:

- **Phased structure**: Group work into sequential phases, each with a
  clear purpose and rationale.
- **One-story-point tasks**: Break each phase into numbered sub-sections
  (1.1, 1.2, etc.) with checkbox task lists.
- **Test-as-you-go ordering**: Order phases and tasks so each part is
  testable without waiting until the end. Reorder from the source PRD
  as needed.
- **Acceptance criteria**: Every sub-section must include verifiable
  acceptance criteria.
- **Testing built in**: Include test creation tasks within each phase,
  not deferred to the end.
- **Database batching**: If database changes are needed, batch related
  migrations and place schema updates in early phases.

### 4. Output

Generate the implementation guide file:

1. Locate the source feature description or PRD document's directory.
2. Create file: `YYYY-MM-DD_implementation-guide.md` in the same
   directory.

For format guidance, see
[references/implementation-guide-format.md](references/implementation-guide-format.md).

Write the guide based on the research and source document. Include
rationale for sequencing decisions and note any deviations from the
source PRD's ordering.
