"""Generate assets/microscope.ico for the desktop shortcut.

Draws a simple flat-style microscope glyph on the app's dark-theme background and
writes a multi-resolution .ico. Pure Pillow, no network. Run once:

    python tools/make_icon.py

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

# Microscope: base, arm, body tube, stage, eyepiece.
# Base
d.rounded_rectangle([70, 196, 186, 214], radius=9, fill=LIGHT)
d.rounded_rectangle([96, 176, 160, 200], radius=6, fill=LIGHT)
# Curved arm (approximate with a thick arc)
d.arc([70, 74, 190, 200], start=110, end=250, fill=ACCENT, width=18)
# Body tube (angled)
d.line([150, 78, 110, 150], fill=ACCENT, width=22)
# Eyepiece
d.rounded_rectangle([150, 56, 178, 92], radius=7, fill=LIGHT)
# Stage
d.rounded_rectangle([84, 150, 150, 162], radius=5, fill=LIGHT)
# Objective / light dot
d.ellipse([100, 150, 120, 170], fill=ACCENT)

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "microscope.ico")
img.save(out_path, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print("Wrote", out_path)
