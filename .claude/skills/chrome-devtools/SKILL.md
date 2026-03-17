---
name: chrome-devtools
description: Browser automation, debugging, and performance analysis using Puppeteer CLI scripts. Use for automating browsers, taking screenshots, analyzing performance, monitoring network traffic, web scraping, form automation, and JavaScript debugging.
license: Apache-2.0
---

# Chrome DevTools Agent Skill

Browser automation via executable Puppeteer scripts. All scripts output JSON for easy parsing.

## Quick Start

**CRITICAL**: Always check `pwd` before running scripts.

### Installation

#### Step 1: Install System Dependencies (Linux/WSL only)

On Linux/WSL, Chrome requires system libraries. Install them first:

```bash
pwd  # Should show current working directory
cd .claude/skills/chrome-devtools/scripts
./install-deps.sh  # Auto-detects OS and installs required libs
```

Supports: Ubuntu, Debian, Fedora, RHEL, CentOS, Arch, Manjaro

**macOS/Windows**: Skip this step (dependencies bundled with Chrome)

#### Step 2: Install Node Dependencies

```bash
npm install  # Installs puppeteer, debug, yargs
```

#### Step 3: Install ImageMagick (Optional, Recommended)

ImageMagick enables automatic screenshot compression to keep files under 5MB:

**macOS:**

```bash
brew install imagemagick
```

**Ubuntu/Debian/WSL:**

```bash
sudo apt-get install imagemagick
```

**Verify:**

```bash
magick -version  # or: convert -version
```

Without ImageMagick, screenshots >5MB will not be compressed (may fail to load in Gemini/Claude).

### Test

```bash
node navigate.js --url https://example.com
# Output: {"success": true, "url": "https://example.com", "title": "Example Domain"}
```

## Available Scripts

All scripts are in `.claude/skills/chrome-devtools/scripts/`

**CRITICAL**: Always check `pwd` before running scripts.

### Script Usage

- `./scripts/README.md`

### Core Automation

- `navigate.js` - Navigate to URLs
- `screenshot.js` - Capture screenshots (full page or element)
- `click.js` - Click elements
- `fill.js` - Fill form fields
- `evaluate.js` - Execute JavaScript in page context

### Analysis & Monitoring

- `snapshot.js` - Extract interactive elements with metadata
- `console.js` - Monitor console messages/errors
- `network.js` - Track HTTP requests/responses
- `performance.js` - Measure Core Web Vitals + record traces

## Usage Patterns

### Single Command

```bash
# Run from PROJECT ROOT - create screenshot dir and run script
# Filename MUST include timestamp (HHMMSS) to prevent overwriting
mkdir -p _screenshots/$(date +%Y-%m-%d)
node .claude/skills/chrome-devtools/scripts/screenshot.js --url https://example.com --output "_screenshots/$(date +%Y-%m-%d)/$(date +%Y-%m-%d_%H%M%S)_example-homepage.jpg" --format jpeg --quality 80 --max-size 0.5
```

**Screenshot Defaults** (unless otherwise instructed):

- **Size**: 1920x1080 viewport (browser default)
- **Format**: JPEG with compression (`--format jpeg --quality 80 --max-size 0.5`)
- **Directory**: `_screenshots/<date>/` at **PROJECT ROOT** (NOT inside .claude/skills/)
- **Date subdirectory**: Create if it doesn't exist (e.g., `2026-01-22`)
- **Filename format**: `<date>_<HHMMSS>_<slug>.jpg` — timestamp is **REQUIRED** to prevent overwriting
  - Example: `2026-01-22_143052_login-form.jpg` (taken at 14:30:52)
  - Use current time: `date +%H%M%S` to get timestamp
- **Slug**: URL/filename-friendly description of the page or topic

### Automatic Image Compression

Screenshots are **automatically compressed** if they exceed 5MB to ensure compatibility with Gemini API and Claude Code (which have 5MB limits). This uses ImageMagick internally:

```bash
# Default: auto-compress if >5MB
node screenshot.js --url https://example.com --output page.png

# Custom size threshold (e.g., 3MB)
node screenshot.js --url https://example.com --output page.png --max-size 3

# Disable compression
node screenshot.js --url https://example.com --output page.png --no-compress
```

**Compression behavior:**

- PNG: Resizes to 90% + quality 85 (or 75% + quality 70 if still too large)
- JPEG: Quality 80 + progressive encoding (or quality 60 if still too large)
- Other formats: Converted to JPEG with compression
- Requires ImageMagick installed (see imagemagick skill)

**Output includes compression info:**

```json
{
  "success": true,
  "output": "/path/to/page.png",
  "compressed": true,
  "originalSize": 8388608,
  "size": 3145728,
  "compressionRatio": "62.50%",
  "url": "https://example.com"
}
```

### Chain Commands (reuse browser)

```bash
# Keep browser open with --close false
node navigate.js --url https://example.com/login --close false
node fill.js --selector "#email" --value "user@example.com" --close false
node fill.js --selector "#password" --value "secret" --close false
node click.js --selector "button[type=submit]"
```

### Parse JSON Output

```bash
# Extract specific fields with jq
node performance.js --url https://example.com | jq '.vitals.LCP'

# Save to file
node network.js --url https://example.com --output /tmp/requests.json
```

## Execution Protocol

### Working Directory Verification

BEFORE executing any script:

1. Verify you are at the **PROJECT ROOT** (not inside .claude/skills/)
2. Run scripts using full path: `node .claude/skills/chrome-devtools/scripts/<script>.js`
3. Save screenshots to `_screenshots/<date>/` at project root

Example:

```bash
pwd  # Should show project root (e.g., /home/user/my-project)
mkdir -p _screenshots/$(date +%Y-%m-%d)
node .claude/skills/chrome-devtools/scripts/screenshot.js --url https://example.com --output "_screenshots/$(date +%Y-%m-%d)/$(date +%Y-%m-%d_%H%M%S)_page.jpg" --format jpeg --quality 80 --max-size 0.5
```

**Note**: Only `cd` into `.claude/skills/chrome-devtools/scripts/` for installation (`npm install`).

### Output Validation

AFTER screenshot/capture operations:

1. Verify file created with `ls -lh <output-path>`
2. Read screenshot using Read tool to confirm content
3. Check JSON output for success:true
4. Report file size and compression status

Example:

```bash
# Set variables for consistent filename across commands
DATE_DIR=$(date +%Y-%m-%d)
FILENAME="${DATE_DIR}_$(date +%H%M%S)_example-homepage.jpg"

mkdir -p "_screenshots/${DATE_DIR}"
node .claude/skills/chrome-devtools/scripts/screenshot.js --url https://example.com --output "_screenshots/${DATE_DIR}/${FILENAME}" --format jpeg --quality 80 --max-size 0.5
ls -lh "_screenshots/${DATE_DIR}/${FILENAME}"  # Verify file exists
# Then use Read tool to visually inspect
```

### Error Recovery

If script fails:

1. Check error message for selector issues
2. Use snapshot.js to discover correct selectors
3. Try XPath selector if CSS selector fails
4. Verify element is visible and interactive

Example:

```bash
# CSS selector fails
node click.js --url https://example.com --selector ".btn-submit"
# Error: waiting for selector ".btn-submit" failed

# Discover correct selector
node snapshot.js --url https://example.com | jq '.elements[] | select(.tagName=="BUTTON")'

# Try XPath
node click.js --url https://example.com --selector "//button[contains(text(),'Submit')]"
```

### Common Mistakes

❌ Wrong working directory → output files go to wrong location
❌ Skipping output validation → silent failures
❌ Using complex CSS selectors without testing → selector errors
❌ Not checking element visibility → timeout errors

✅ Always verify `pwd` before running scripts
✅ Always validate output after screenshots
✅ Use snapshot.js to discover selectors
✅ Test selectors with simple commands first

## Common Workflows

### Web Scraping

```bash
node evaluate.js --url https://example.com --script "
  Array.from(document.querySelectorAll('.item')).map(el => ({
    title: el.querySelector('h2')?.textContent,
    link: el.querySelector('a')?.href
  }))
" | jq '.result'
```

### Performance Testing

```bash
PERF=$(node performance.js --url https://example.com)
LCP=$(echo $PERF | jq '.vitals.LCP')
if (( $(echo "$LCP < 2500" | bc -l) )); then
  echo "✓ LCP passed: ${LCP}ms"
else
  echo "✗ LCP failed: ${LCP}ms"
fi
```

### Form Automation

```bash
node fill.js --url https://example.com --selector "#search" --value "query" --close false
node click.js --selector "button[type=submit]"
```

### Error Monitoring

```bash
node console.js --url https://example.com --types error,warn --duration 5000 | jq '.messageCount'
```

## Script Options

All scripts support:

- `--headless false` - Show browser window
- `--close false` - Keep browser open for chaining
- `--timeout 30000` - Set timeout (milliseconds)
- `--wait-until networkidle2` - Wait strategy

See `./scripts/README.md` for complete options.

## Output Format

All scripts output JSON to stdout:

```json
{
  "success": true,
  "url": "https://example.com",
  ... // script-specific data
}
```

Errors go to stderr:

```json
{
  "success": false,
  "error": "Error message"
}
```

## Finding Elements

Use `snapshot.js` to discover selectors:

```bash
node snapshot.js --url https://example.com | jq '.elements[] | {tagName, text, selector}'
```

## Troubleshooting

### Common Errors

**"Cannot find package 'puppeteer'"**

- Run: `npm install` in the scripts directory

**"error while loading shared libraries: libnss3.so"** (Linux/WSL)

- Missing system dependencies
- Fix: Run `./install-deps.sh` in scripts directory
- Manual install: `sudo apt-get install -y libnss3 libnspr4 libasound2t64 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1`

**"Failed to launch the browser process"**

- Check system dependencies installed (Linux/WSL)
- Verify Chrome downloaded: `ls ~/.cache/puppeteer`
- Try: `npm rebuild` then `npm install`

**Chrome not found**

- Puppeteer auto-downloads Chrome during `npm install`
- If failed, manually trigger: `npx puppeteer browsers install chrome`

### Script Issues

**Element not found**

- Get snapshot first to find correct selector: `node snapshot.js --url <url>`

**Script hangs**

- Increase timeout: `--timeout 60000`
- Change wait strategy: `--wait-until load` or `--wait-until domcontentloaded`

**Blank screenshot**

- Wait for page load: `--wait-until networkidle2`
- Increase timeout: `--timeout 30000`

**Permission denied on scripts**

- Make executable: `chmod +x *.sh`

**Screenshot too large (>5MB)**

- Install ImageMagick for automatic compression
- Manually set lower threshold: `--max-size 3`
- Use JPEG format instead of PNG: `--format jpeg --quality 80`
- Capture specific element instead of full page: `--selector .main-content`

**Compression not working**

- Verify ImageMagick installed: `magick -version` or `convert -version`
- Check file was actually compressed in output JSON: `"compressed": true`
- For very large pages, use `--selector` to capture only needed area

## Reference Documentation

Detailed guides available in `./references/`:

- [CDP Domains Reference](./references/cdp-domains.md) - 47 Chrome DevTools Protocol domains
- [Puppeteer Quick Reference](./references/puppeteer-reference.md) - Complete Puppeteer API patterns
- [Performance Analysis Guide](./references/performance-guide.md) - Core Web Vitals optimization

## Advanced Usage

### Custom Scripts

Create custom scripts using shared library:

```javascript
import { getBrowser, getPage, closeBrowser, outputJSON } from './lib/browser.js';
// Your automation logic
```

### Direct CDP Access

```javascript
const client = await page.createCDPSession();
await client.send('Emulation.setCPUThrottlingRate', { rate: 4 });
```

See reference documentation for advanced patterns and complete API coverage.

## Project-Specific Notes

### Dashboard Verification Workflow

**Always read the port from `project.config.json`** — never hardcode it. Dev server: `pnpm dev`.

```bash
# Read port first
PORT=$(node -p "require('./project.config.json').port")

# 1. Screenshot the page
mkdir -p _screenshots/$(date +%Y-%m-%d)
node .claude/skills/chrome-devtools/scripts/screenshot.js \
  --url "http://localhost:${PORT}/chat" \
  --output "_screenshots/$(date +%Y-%m-%d)/$(date +%Y-%m-%d_%H%M%S)_chat.jpg" \
  --format jpeg --quality 80 --max-size 0.5

# 2. Check for console errors
node .claude/skills/chrome-devtools/scripts/console.js \
  --url "http://localhost:${PORT}/chat" --types error --duration 3000
```

### Multi-Step Interactions (evaluate + screenshot)

**The `--close false` chaining approach is broken for screenshots.** `screenshot.js` without `--url` connects to `about:blank`, not the tab opened by a previous `evaluate.js --close false`. This means you CANNOT: evaluate → screenshot across separate script calls.

**For multi-step workflows (inject state, interact, then screenshot), write a custom `.mjs` script:**

```javascript
// _screenshots/my-test.mjs
import { createRequire } from 'module';
const require = createRequire('<PROJECT_ROOT>/.claude/skills/chrome-devtools/scripts/package.json');
const puppeteer = require('puppeteer');

const browser = await puppeteer.launch({ headless: true });
const page = await browser.newPage();
await page.setViewport({ width: 1440, height: 900 });
const config = require('<PROJECT_ROOT>/project.config.json');
await page.goto(`http://localhost:${config.port}/chat`, { waitUntil: 'networkidle2', timeout: 10000 });

// Interact with the page
await page.evaluate(() => {
  // DOM manipulation, click simulation, etc.
});
await new Promise(r => setTimeout(r, 300)); // wait for layout

// Mouse interactions (hover, scroll)
await page.mouse.move(700, 400);
await page.mouse.wheel({ deltaY: 200 });
await new Promise(r => setTimeout(r, 100));

// Screenshot
await page.screenshot({
  path: '<PROJECT_ROOT>/_screenshots/2026-01-01/result.jpg',
  type: 'jpeg',
  quality: 95
});

await browser.close();
```

**Key details for custom scripts:**
- Puppeteer is installed in `.claude/skills/chrome-devtools/scripts/node_modules/`, NOT the project root. Use `createRequire()` pointed at the skill's `package.json` to import it.
- Use **absolute paths** for screenshot output — Puppeteer resolves relative paths from the script's location, not the project root.
- Clean up custom test scripts after use.

### evaluate.js Gotchas

- **No `return` statements** — the script runs inside `page.evaluate()`. Use the expression result or `JSON.stringify(...)` as the last statement.
- **No `await`** — the script body is not async. Use synchronous patterns or callbacks.
- **Multi-line scripts with special characters may fail silently.** For complex scripts, write to a file and pass via variable: `SCRIPT=$(cat mytest.js) && node evaluate.js --script "$SCRIPT"`

### Headless Chrome Limitations

- **macOS overlay scrollbars are invisible in screenshots.** Overlay scrollbars only appear during active scrolling and fade immediately. Even `page.mouse.wheel()` + short delay won't capture them. Use `evaluate.js` to check computed styles instead (`scrollbarColor`, `scrollbarWidth`).
- **DOM injection bypasses React.** Injecting elements directly into the DOM won't trigger React state updates, layout effects, or library behavior (e.g., `use-stick-to-bottom`). You may need to manually set inline styles (like `overflow: auto`) that would normally be set by React lifecycle hooks.
- **Headless Chrome uses classic (non-overlay) scrollbars** regardless of macOS system settings. Visual behavior may differ from the user's real browser.

### Selector Tips

- `:has-text()` is **not valid CSS** in Puppeteer — use XPath instead: `//button[contains(., 'New Job')]`
- Use `snapshot.js` to discover available selectors before writing click/fill commands
- Radix UI components generate dynamic IDs — prefer `data-*` attributes or text-based XPath selectors

## External Resources

- [Puppeteer Documentation](https://pptr.dev/)
- [Chrome DevTools Protocol](https://chromedevtools.github.io/devtools-protocol/)
- [Scripts README](./scripts/README.md)
