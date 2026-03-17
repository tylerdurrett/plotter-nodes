---
name: new-video
description: Guide users through creating new Remotion video compositions in the @repo/video package. Use when users want to create a new video, animation, or composition, or when they invoke /new-video. Handles scaffolding, explains Remotion concepts, and helps with composition editing.
---

# New Video Composition

Create a new Remotion composition in `packages/video/`.

## Process

### 1. Gather Requirements

Ask the user what they want to create. Determine:

- **Name**: PascalCase composition name (e.g., `LogoReveal`, `ProductDemo`)
- **Description**: What the video should show or do

If the user is non-technical, briefly explain:
- A composition is a React component that renders video frame by frame
- `useCurrentFrame()` returns the current frame number (like a position on a timeline)
- `fps` is frames per second (30 is standard — so 90 frames = 3 seconds)
- You animate by changing styles based on the frame number

### 2. Scaffold

Run the scaffolding script:

```bash
pnpm new-video <CompositionName>
```

This creates `packages/video/src/compositions/<Name>.tsx` with a starter
template, registers it in `Root.tsx`, and adds it to barrel exports.

### 3. Customize

Open `packages/video/src/compositions/<Name>.tsx` and implement the
user's vision. The template includes working imports for all key APIs.

**Common animation patterns:**

Fade in:
```tsx
const opacity = interpolate(frame, [0, 30], [0, 1], {
  extrapolateRight: 'clamp',
});
```

Spring physics:
```tsx
const scale = spring({ frame, fps, config: { damping: 12 } });
```

Delayed entrance (element appears at frame 30):
```tsx
import { Sequence } from 'remotion';
<Sequence from={30}><MyElement /></Sequence>
```

Staggered animations:
```tsx
const item1 = spring({ frame, fps });
const item2 = spring({ frame: frame - 10, fps });
const item3 = spring({ frame: frame - 20, fps });
```

Color/position interpolation:
```tsx
const x = interpolate(frame, [0, 60], [0, 200]);
```

**Tailwind works in compositions** — use any Tailwind classes for
styling. They render in both Studio and the web Player.

**Reference**: The `HelloWorld` composition at
`packages/video/src/compositions/HelloWorld.tsx` demonstrates all core
APIs and is a good starting point to study.

### 4. Preview

Tell the user to run:

```bash
pnpm studio
```

Studio opens in the browser with a sidebar listing all compositions.
Select the new one to preview. Studio hot-reloads on file save.

### 5. Embed in Web App (Optional)

If the user wants to embed the video in the Next.js app, follow the
pattern in `apps/web/src/app/video/`:

1. Create a route directory (e.g., `apps/web/src/app/my-video/`)
2. Create a `'use client'` component that imports `Player` from
   `@remotion/player` and the composition from `@repo/video`
3. Create a `page.tsx` server component that renders it

See `apps/web/src/app/video/video-player.tsx` for the exact pattern.
