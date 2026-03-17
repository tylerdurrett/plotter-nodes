---
name: ui-dev
description: Guidelines for building UI components and features. Use automatically when creating or modifying UI code in this project — components, pages, layouts, or styling. Ensures consistency with project conventions.
---

# UI Development

Follow these conventions when building any UI in this project.

## Before Writing UI Code

1. Read `docs/ui-conventions.md` for the full decision rules
2. Check `packages/ui/src/components/` for existing components that might fit
3. Check if Shadcn has a component you need: `pnpm dlx shadcn@latest add <component> --cwd packages/ui`

## Quick Rules

**Componentization:** Inline Tailwind until the third use, or until it's a semantic concept with behavior. Don't preemptively extract.

**Tokens:** Use Tailwind values directly. Only create CSS custom properties when a semantic meaning is clearly repeating across multiple components.

**Components go in:**
- `packages/ui/src/components/` → reusable across the app (export from `@repo/ui`)
- Colocated with feature → feature-specific only

**Performance:**
- Optimistic updates via TanStack Query
- Only animate `transform` and `opacity` (GPU-composited)
- Never animate layout properties (width, height, margins)

**Theme:** Dark-first. Don't build light mode unless requested.

**Scrollbars:** Styled globally — don't add custom scrollbar styles per-component.

**Responsive:** Use container queries when a component lives in varying-width contexts. Don't add speculatively.

## When You're Done

- If you extracted a new reusable component to `@repo/ui`, export it from the package
- If you created a new semantic token, add a comment in the CSS explaining what it means
- Run `pnpm lint` and `pnpm build` to verify
