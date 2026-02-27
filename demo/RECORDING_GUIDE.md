# Terradev + Claude Code Demo Recording Guide

## Prerequisites

```bash
brew install asciinema agg gifsicle
```

## Step 1: Record Terminal Session (~5 min)

```bash
cd ~/CascadeProjects/Terradev
asciinema rec demo/terradev-demo.cast --cols 100 --rows 30
```

### Demo Script (run these commands inside the recording)

```bash
# 1. Show Claude Code MCP integration
claude mcp list

# 2. Live GPU pricing across 11+ providers
terradev quote -g H100

# 3. Dry-run cluster provisioning
terradev provision -g A100 --dry-run

# 4. Show tier info
terradev upgrade --show

# 5. Provision 4x A100s (or dry-run if you don't want real infra)
terradev provision -g A100 -n 4 --dry-run

# 6. Status check
terradev status
```

Press **Ctrl+D** when done.

## Step 2: Convert to GIF (~2 min)

```bash
agg demo/terradev-demo.cast demo/terradev-claude.gif \
  --font-family "JetBrains Mono" \
  --font-size 14 \
  --theme dracula \
  --cols 100 \
  --rows 30 \
  --fps 10
```

## Step 3: Optimize & Verify

```bash
# Compress (target <5MB for GitHub/PyPI)
gifsicle -O3 --lossy=80 -D -H demo/terradev-claude.gif -o demo/terradev-claude-opt.gif

# Verify size
ls -lh demo/terradev-claude-opt.gif
```

## Step 4: Ship

GIF goes into repo root and READMEs reference it:

```bash
cp demo/terradev-claude-opt.gif ./terradev-claude-opt.gif
git add terradev-claude-opt.gif
git commit -m "Add Claude Code + Terradev demo GIF"
git push
```

## Recording Tips

- **80-100 chars width** — terminal standard, no horizontal scroll
- **14px font** — crisp on HiDPI
- **Dracula theme** — matches most dev terminals
- **10fps** — smooth but lightweight file size
- **Pause 1-2s between commands** — lets viewers read output
- **Trim first/last 2s** — no blank scrolling at start/end
