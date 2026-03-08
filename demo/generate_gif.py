#!/usr/bin/env python3
"""
Generate Terradev demo GIF purely with Python + Pillow.
No brew, no asciinema, no agg needed.

Commands are typed word-by-word. Output streams line-by-line.

Usage:
    python demo/generate_gif.py
    # Output: demo/terradev-demo.gif
"""

import os
from PIL import Image, ImageDraw, ImageFont

# ── Configuration ────────────────────────────────────────────────────────

CHAR_W = 10        # px per character (monospace)
CHAR_H = 20        # px per line
PAD_X = 24         # left/right padding
PAD_Y = 20         # top/bottom padding
COLS = 92          # terminal width in chars
MAX_LINES = 30     # fixed terminal height so all frames are the same size
BG = (30, 30, 46)  # Catppuccin Mocha background
FG = (205, 214, 244)
GREEN = (166, 227, 161)
YELLOW = (249, 226, 175)
RED = (243, 139, 168)
CYAN = (137, 220, 235)
BLUE = (137, 180, 250)
MAGENTA = (203, 166, 247)
DIM = (108, 112, 134)
ORANGE = (250, 179, 135)
TITLE_BAR = (24, 24, 37)
CURSOR_COLOR = (205, 214, 244)  # blinking block cursor

IMG_W = COLS * CHAR_W + PAD_X * 2
IMG_H = MAX_LINES * CHAR_H + PAD_Y * 2 + 32

# Try to find a monospace font
FONT = None
FONT_PATHS = [
    "/System/Library/Fonts/SFMono-Regular.otf",
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.dfont",
    "/Library/Fonts/SF-Mono-Regular.otf",
]
for fp in FONT_PATHS:
    if os.path.exists(fp):
        try:
            FONT = ImageFont.truetype(fp, 14)
            break
        except Exception:
            continue
if FONT is None:
    FONT = ImageFont.load_default()


# ── Frame Renderer ───────────────────────────────────────────────────────

def render_frame(lines, title="Terminal — terradev", cursor_pos=None):
    """
    Render a terminal frame at fixed size.
    lines: list of items, each either:
      - (text, color)            — single-color line
      - [(text, color), ...]     — multi-segment line
    cursor_pos: (line_idx, char_offset) to draw a block cursor, or None.
    """
    img = Image.new("RGB", (IMG_W, IMG_H), BG)
    draw = ImageDraw.Draw(img)

    # Title bar
    draw.rectangle([0, 0, IMG_W, 30], fill=TITLE_BAR)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        draw.ellipse([12 + i * 22, 9, 24 + i * 22, 21], fill=c)
    if title:
        draw.text((IMG_W // 2 - len(title) * 3, 7), title, fill=DIM, font=FONT)

    y_off = 32 + PAD_Y
    for line in lines:
        x = PAD_X
        if isinstance(line, tuple):
            text, color = line
            draw.text((x, y_off), text, fill=color, font=FONT)
        elif isinstance(line, list):
            for text, color in line:
                draw.text((x, y_off), text, fill=color, font=FONT)
                x += len(text) * CHAR_W
        y_off += CHAR_H

    # Draw block cursor
    if cursor_pos is not None:
        cy, cx = cursor_pos
        cx_px = PAD_X + cx * CHAR_W
        cy_px = 32 + PAD_Y + cy * CHAR_H
        draw.rectangle(
            [cx_px, cy_px, cx_px + CHAR_W, cy_px + CHAR_H],
            fill=CURSOR_COLOR,
        )

    return img


# ── Typing helpers ───────────────────────────────────────────────────────

def type_command(existing_lines, prompt_segs, cmd_text, word_delay=120):
    """
    Generate frames that type out `cmd_text` word-by-word after the prompt.
    Returns (frames, durations, final_lines).
    """
    frames = []
    durations = []
    words = cmd_text.split(" ")

    typed_so_far = ""
    prompt_char_len = sum(len(t) for t, _ in prompt_segs)

    for i, word in enumerate(words):
        if i > 0:
            typed_so_far += " "
        typed_so_far += word

        line = list(prompt_segs) + [(typed_so_far, FG)]
        cur_lines = existing_lines + [line]
        cursor_col = prompt_char_len + len(typed_so_far)
        frames.append(render_frame(cur_lines, cursor_pos=(len(cur_lines) - 1, cursor_col)))
        durations.append(word_delay)

    # Final hold with cursor
    final_line = list(prompt_segs) + [(cmd_text, FG)]
    final_lines = existing_lines + [final_line]
    frames.append(render_frame(final_lines, cursor_pos=(len(final_lines) - 1, prompt_char_len + len(cmd_text))))
    durations.append(400)

    return frames, durations, final_lines


def stream_output(existing_lines, output_lines, line_delay=80):
    """
    Generate frames that reveal output_lines one line at a time.
    Returns (frames, durations, final_lines).
    """
    frames = []
    durations = []
    cur = list(existing_lines)

    for line in output_lines:
        cur = cur + [line]
        frames.append(render_frame(cur))
        durations.append(line_delay)

    return frames, durations, cur


def hold(lines, ms):
    """Single hold frame."""
    return [render_frame(lines)], [ms]


# ── Demo Scenes ──────────────────────────────────────────────────────────

def build_frames():
    all_frames = []
    all_durations = []

    def add(f, d):
        all_frames.extend(f)
        all_durations.extend(d)

    prompt = [("❯ ", GREEN)]

    # ═══════════════════════════════════════════════════════════════════
    # Scene 1: terradev quote -g H100
    # ═══════════════════════════════════════════════════════════════════

    # Type the command word by word
    f, d, lines = type_command([], prompt, "terradev quote -g H100", word_delay=140)
    add(f, d)

    # Blank line + scanning message
    output1 = [
        ("", FG),
        [("🔍 ", FG), ("Scanning 15 providers in parallel...", CYAN)],
    ]
    f, d, lines = stream_output(lines, output1, line_delay=600)
    add(f, d)

    # Pause for "scanning"
    add(*hold(lines, 800))

    # Results stream in
    header = "Provider        GPU    $/hr     Region        Spot   Score"
    sep = "─" * len(header)
    table_lines = [
        [("✅ ", FG), ("9 quotes retrieved in 1.8s", GREEN)],
        ("", FG),
        (header, YELLOW),
        (sep, DIM),
        ("RunPod          H100   $2.49    US-TX         ✓      0.94", GREEN),
        ("Vast.ai         H100   $2.69    US-Central    ✓      0.91", GREEN),
        ("TensorDock      H100   $2.89    US-East       ✓      0.89", FG),
        ("CoreWeave       H100   $3.04    LAS1          ✓      0.87", FG),
        ("Lambda          H100   $3.29    us-west-1     ✗      0.85", FG),
        ("Hyperstack      H100   $3.49    US-East       ✓      0.83", DIM),
        ("AWS             H100   $4.68    us-east-1     ✓      0.79", DIM),
        ("GCP             H100   $5.07    us-central1   ✓      0.76", DIM),
        ("Azure           H100   $5.12    eastus        ✓      0.74", DIM),
        (sep, DIM),
        ("", FG),
        [("💰 ", FG), ("Best: $2.49/hr (RunPod) ", GREEN), ("— saves 51% vs Azure", YELLOW)],
        [("⚡ ", FG), ("Spot available on 7/9 providers", CYAN)],
    ]
    f, d, lines = stream_output(lines, table_lines, line_delay=70)
    add(f, d)

    # Hold on the full table
    add(*hold(lines, 3000))

    # ═══════════════════════════════════════════════════════════════════
    # Scene 2: terradev provision -g H100
    # ═══════════════════════════════════════════════════════════════════

    # Fresh screen
    f, d, lines = type_command([], prompt, "terradev provision -g H100", word_delay=140)
    add(f, d)

    # Spinner
    spinner_out = [
        ("", FG),
        [("⠋ ", CYAN), ("Provisioning H100 on RunPod (US-TX)...", FG)],
    ]
    f, d, lines = stream_output(lines, spinner_out, line_delay=400)
    add(f, d)
    add(*hold(lines, 1000))

    # Result
    prov_result = [
        ("", FG),
        [("✅ ", FG), ("Provisioned!", GREEN)],
        ("", FG),
        [("   Instance:  ", DIM), ("rpd-h100-7x9k2f", FG)],
        [("   GPU:       ", DIM), ("1× NVIDIA H100 80GB SXM", FG)],
        [("   Region:    ", DIM), ("US-TX", FG)],
        [("   Price:     ", DIM), ("$2.49/hr (spot)", GREEN)],
        [("   SSH:       ", DIM), ("ssh root@207.148.5.42", CYAN)],
    ]
    f, d, lines = stream_output(lines, prov_result, line_delay=90)
    add(f, d)
    add(*hold(lines, 3000))

    # ═══════════════════════════════════════════════════════════════════
    # Scene 3: terradev inference topology
    # ═══════════════════════════════════════════════════════════════════

    f, d, lines = type_command([], prompt, "terradev inference topology --arch mi300x", word_delay=130)
    add(f, d)

    topo_out = [
        ("", FG),
        [("🔬 ", FG), ("GPU NUMA Topology — AMD MI300X", MAGENTA)],
        ("", FG),
        ("   Architecture:  CDNA3 chiplet", FG),
        [("   XCDs:          ", DIM), ("8 Accelerated Compute Dies", ORANGE)],
        [("   HBM:           ", DIM), ("192 GB HBM3 (8 stacks × 24 GB)", FG)],
        [("   L2 per XCD:    ", DIM), ("4 MB (32 MB total)", FG)],
        [("   Intra-GPU NUMA:", DIM), (" enabled", GREEN)],
        ("", FG),
        [("   XCD-Aware Env Vars ", YELLOW), ("(apply to vLLM/SGLang):", DIM)],
        ("   ┌─────────────────────────────────────────────┐", DIM),
        [("   │ ", DIM), ("AITER_XCD_AWARE_ATTENTION=1", CYAN), ("               │", DIM)],
        [("   │ ", DIM), ("CK_BLOCK_MAPPING_POLICY=xcd_aware", CYAN), ("         │", DIM)],
        [("   │ ", DIM), ("NCCL_INTRA_GPU_NUMA=1", CYAN), ("                     │", DIM)],
        [("   │ ", DIM), ("HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7", CYAN), ("      │", DIM)],
        ("   └─────────────────────────────────────────────┘", DIM),
    ]
    f, d, lines = stream_output(lines, topo_out, line_delay=80)
    add(f, d)
    add(*hold(lines, 3000))

    # ═══════════════════════════════════════════════════════════════════
    # Scene 4: Closing tagline
    # ═══════════════════════════════════════════════════════════════════

    tag_lines = [
        ("", FG),
        ("", FG),
        ("", FG),
        ("", FG),
        [("   terradev", CYAN), (" — Multi-cloud GPU provisioning CLI", FG)],
        ("", FG),
        [("   ", FG), ("pip install terradev-cli", GREEN)],
        ("", FG),
        [("   15 clouds", YELLOW), (" · ", DIM), ("NUMA topology", MAGENTA),
         (" · ", DIM), ("price intelligence", ORANGE)],
        [("   disaggregated inference", BLUE), (" · ", DIM), ("MoE templates", CYAN)],
        ("", FG),
        [("   github.com/theoddden/Terradev", DIM)],
        ("", FG),
    ]
    f, d, _ = stream_output([], tag_lines, line_delay=120)
    add(f, d)
    add(*hold(tag_lines, 3500))

    return all_frames, all_durations


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    print("Generating frames...")
    frames, durations = build_frames()
    print(f"  {len(frames)} frames built")

    out_path = os.path.join(out_dir, "terradev-demo.gif")
    print(f"Saving GIF to {out_path}...")

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )

    size_kb = os.path.getsize(out_path) / 1024
    print(f"✅ Done! {out_path} ({size_kb:.0f} KB)")

    if size_kb > 5000:
        print("⚠️  GIF is >5MB — consider reducing frame count or resolution")
    else:
        print("✅ Size is under 5MB — good for GitHub README")


if __name__ == "__main__":
    main()
