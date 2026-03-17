---
name: Read library source before proposing workarounds
description: When something doesn't work with a library, read the actual source code to understand why before changing the implementation approach
type: feedback
---

When a library API doesn't behave as expected, read the library's source code and trace the actual code path before proposing alternative architectures or workarounds.

**Why:** I assumed Leva's `onChange` was "unreliable in certain configurations" and proposed switching to a direct store subscription approach — a significantly different and more complex architecture. The real issue was simply passing `onChange` in the wrong place (settings object vs. schema entries). Reading `parseOptions()` in Leva's source made this immediately clear, and the fix was a small, targeted change that kept the original simple design.

**How to apply:** When a library call doesn't work:
1. Read the library source code (usually in `node_modules/<pkg>/dist/`) and trace the exact code path
2. Understand *why* it's not working before concluding the library is at fault
3. Never propose an architectural change as a workaround without first proving the simpler approach can't work
4. "Known to be unreliable" is not an acceptable explanation — prove it with source code evidence
