---
name: feature-request
description: Facilitate scoping and discovery for new feature initiatives through conversational research and questioning. Use when the user wants to propose, discuss, or scope a new feature idea, or when they invoke the /feature-request command. Outputs a feature description document to _tasks/_planning/.
---

# Feature Request

Scope a new feature initiative through focused discovery conversation,
then output a structured feature description document.

## Process

### 1. Research

Ground in the codebase and any relevant external systems. Understand
existing architecture, conventions, and related functionality before
asking questions.

### 2. Discovery Conversation

Ask the user targeted questions to clarify scope. Focus on:

- **End-user capabilities**: What can the user do when this is complete?
- **Boundaries**: What is explicitly in scope vs out of scope?
- **Architecture implications**: How does this fit with existing systems?
- **Risks and trade-offs**: What constraints or concerns exist?

Iterate until the user confirms the scope is well-defined.

### 3. Output

When the user is satisfied with the scope, generate the feature description:

1. Create folder: `_tasks/_planning/YYYY-MM-DD_initiative-slug/`
2. Create file: `YYYY-MM-DD_feature-description.md`

For format guidance, see
[references/feature-description-format.md](references/feature-description-format.md).

Write the description based on the conversation, including rationale for
decisions. Include all relevant detail but omit code unless specifically
discussed.
