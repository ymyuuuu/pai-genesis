"""Compose the submission video (1920x1080, 50 fps) from text cards, side-by-side clips and result figures.

Usage: python compose_video.py spec.json out.mp4
spec.json is a list of segments:
  {"type": "text", "lines": ["title", "line", ...], "seconds": 5}
  {"type": "pair", "left": "videos/a.mp4", "right": "videos/b.mp4", "caption": "..."}
  {"type": "image", "path": "results/fig.png", "caption": "...", "seconds": 8}
"""
import json
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 50
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_REG = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BG = (18, 24, 38)


def canvas():
    return Image.new("RGB", (W, H), BG)


def draw_caption(img, text, y=40, size=44):
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, size)
    tw = d.textlength(text, font=f)
    d.text(((W - tw) / 2, y), text, font=f, fill="white")


def text_card(lines, seconds):
    img = canvas()
    d = ImageDraw.Draw(img)
    title, body = lines[0], lines[1:]
    ft, fb = ImageFont.truetype(FONT, 64), ImageFont.truetype(FONT_REG, 40)
    y = 260 if body else 480
    d.text(((W - d.textlength(title, font=ft)) / 2, y), title, font=ft, fill="white")
    y += 130
    for line in body:
        d.text(((W - d.textlength(line, font=fb)) / 2, y), line, font=fb, fill=(215, 222, 235))
        y += 64
    frame = np.asarray(img)
    return [frame] * int(seconds * FPS)


def image_card(path, caption, seconds):
    img = canvas()
    fig = Image.open(path).convert("RGB")
    scale = min((W - 80) / fig.width, (H - 180) / fig.height)
    fig = fig.resize((int(fig.width * scale), int(fig.height * scale)), Image.LANCZOS)
    img.paste(fig, ((W - fig.width) // 2, 140 + (H - 160 - fig.height) // 2))
    draw_caption(img, caption)
    frame = np.asarray(img)
    return [frame] * int(seconds * FPS)


def pair(left, right, caption):
    a = [f for f in imageio.get_reader(left)]
    b = [f for f in imageio.get_reader(right)]
    n = max(len(a), len(b))
    a += [a[-1]] * (n - len(a))
    b += [b[-1]] * (n - len(b))
    frames = []
    base = canvas()
    draw_caption(base, caption, y=120, size=48)
    for fa, fb in zip(a, b):
        img = base.copy()
        img.paste(Image.fromarray(fa).resize((940, 529)), (13, 300))
        img.paste(Image.fromarray(fb).resize((940, 529)), (967, 300))
        frames.append(np.asarray(img))
    return frames


def main():
    spec_path, out = sys.argv[1], sys.argv[2]
    with open(spec_path) as f:
        spec = json.load(f)
    writer = imageio.get_writer(out, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
    total = 0
    for seg in spec:
        if seg["type"] == "text":
            frames = text_card(seg["lines"], seg["seconds"])
        elif seg["type"] == "image":
            frames = image_card(seg["path"], seg["caption"], seg["seconds"])
        else:
            frames = pair(seg["left"], seg["right"], seg["caption"])
        for fr in frames:
            writer.append_data(fr)
        total += len(frames)
    writer.close()
    print(f"wrote {out}: {total / FPS:.1f} s")


if __name__ == "__main__":
    main()
