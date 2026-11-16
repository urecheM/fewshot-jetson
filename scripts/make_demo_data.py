#!/usr/bin/env python3
"""
make_demo_data.py - draw placeholder images so the pipeline can be tried before any photos taken
"""

import argparse
import os
import random
import sys

from PIL import Image, ImageDraw

CLASSES = {
    "mug": ("rectangle", (200, 60, 50)),
    "water_bottle": ("tall", (50, 110, 210)),
    "pen": ("line", (40, 160, 70)),
}


def draw_one(shape, colour, rng, size=320):
    """One placehold image-random grey background plus a shape."""
    grey = rng.randint(90, 200)
    img = Image.new("RGB", (size, size), (grey, grey, grey))
    d = ImageDraw.Draw(img)
    # Background clutter (so images within a class are not identical)
    for _ in range(40):
        x, y = rng.randint(0, size), rng.randint(0, size)
        g = rng.randint(60, 230)
        d.ellipse([x, y, x + rng.randint(2, 12), y + rng.randint(2, 12)], fill=(g, g, g))
    cx, cy = size // 2 + rng.randint(-40, 40), size // 2 + rng.randint(-40, 40)
    s = rng.randint(50, 80)
    c = tuple(max(0, min(255, v + rng.randint(-25, 25))) for v in colour)
    if shape == "rectangle":
        d.rectangle([cx - s, cy - s, cx + s, cy + s], fill=c)
        d.arc([cx + s - 10, cy - s // 2, cx + s + 40, cy + s // 2], 270, 90, fill=c, width=10)
    elif shape == "tall":
        d.rectangle([cx - s // 2, cy - 2 * s + 20, cx + s // 2, cy + s], fill=c)
        d.rectangle([cx - s // 4, cy - 2 * s, cx + s // 4, cy - 2 * s + 20], fill=(30, 30, 30))
    else:
        d.line([cx - 2 * s, cy + s, cx + 2 * s, cy - s], fill=c, width=14)
    return img


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="demo_data")
    parser.add_argument("--support-per-class", type=int, default=5)
    parser.add_argument("--query-per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    for split, count in (("support", args.support_per_class), ("query", args.query_per_class)):
        for name, (shape, colour) in CLASSES.items():
            folder = os.path.join(args.out, split, name)
            os.makedirs(folder, exist_ok=True)
            for i in range(count):
                draw_one(shape, colour, rng).save(os.path.join(folder, "%s_%02d.jpg" % (name, i)))
    print("Wrote placeholder images to %s/support and %s/query" % (args.out, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
