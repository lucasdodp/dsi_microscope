"""Generate assets/image_lab.ico for the Image Lab desktop shortcut.

Draws a flat-style "stacked images" glyph (a z-stack of frames, which is what
the Image Lab works on) on the app's dark-theme background, so its shortcut is
visually distinct from the microscope one. Pure Pillow, no network. Run once:

    python tools/make_image_lab_icon.py

Re-run only if you want to change the icon; the committed .ico is what the
shortcut points at.
"""

import os

from PIL import Image, ImageDraw

# App dark-theme accent (matches config.STYLESHEET feel): teal on charcoal.
BG = (30, 33, 40)
ACCENT = (56, 189, 176)
LIGHT = (220, 226, 232)

# Draw large, then downsample for crisp edges.
S = 256
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

# Rounded-square background.
d.rounded_rectangle([8, 8, S - 8, S - 8], radius=44, fill=BG)

# A z-stack: three offset image frames, back to front. The offset is what reads
# as depth — the same stack the 3-D volume view turns.
frames = [(56, 92), (78, 70), (100, 48)]  # (x, y) top-left of each 108x108 frame
size = 108
for i, (fx, fy) in enumerate(frames):
    fill = BG if i < len(frames) - 1 else BG
    outline = LIGHT if i == len(frames) - 1 else ACCENT
    d.rounded_rectangle([fx, fy, fx + size, fy + size], radius=12,
                        fill=fill, outline=outline, width=6)

# A couple of "beads" on the front frame, to say what the images hold.
fx, fy = frames[-1]
d.ellipse([fx + 34, fy + 30, fx + 52, fy + 48], fill=ACCENT)
d.ellipse([fx + 62, fy + 60, fx + 74, fy + 72], fill=LIGHT)

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "image_lab.ico")
img.save(out_path, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print("Wrote", out_path)
