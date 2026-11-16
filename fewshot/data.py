"""
data.py - finding images on disk and turning them into network input.
"""

import os
import random

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_images_by_class(root):
    """Scan `root` and return {class_name: [sorted image paths]}.

    Sorting makes runs repeatable. The same folder always yields the same
    order, so the same images are chosen as "shots" for a given seed.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError("Image folder not found: %s" % root)
    result = {}
    for class_name in sorted(os.listdir(root)):
        class_dir = os.path.join(root, class_name)
        if not os.path.isdir(class_dir) or class_name.startswith("."):
            continue
        paths = [
            os.path.join(class_dir, f)
            for f in sorted(os.listdir(class_dir))
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]
        if paths:
            result[class_name] = paths
    if not result:
        raise ValueError(
            "No images found under %s. Expected sub-folders named after each "
            "object (e.g. mug/, water_bottle/, pen/) containing .jpg/.png files." % root
        )
    return result


def select_shots(paths_by_class, shots, seed=0):
    """Pick `shots` example images per class (all of them if shots <= 0).

    A fixed random seed means "5-shot" selects the same five images on every
    run and for every backbone, so model comparisons are like-for-like.
    """
    rng = random.Random(seed)
    selected = {}
    for class_name, paths in paths_by_class.items():
        if shots and shots > 0 and len(paths) > shots:
            selected[class_name] = sorted(rng.sample(paths, shots))
        else:
            selected[class_name] = list(paths)
    return selected


def load_image(path):
    """Open an image file and force 3-channel RGB (drops alpha, expands greyscale)."""
    with Image.open(path) as im:
        return im.convert("RGB")


def preprocess(pil_image, img_size, mean, std, resize_ratio=256.0 / 224.0):
    """Convert a PIL image into the array for backbone.

    Steps:
      1. Resize so the SHORTER side is slightly larger than the target
         (256 for a 224 target), keeping aspect ratio
      2. Centre-crop a square of img_size x img_size
      3. Scale pixel values from 0..255 to 0..1
      4. Subtract the mean and divide by the standard deviation per channel,
         matching the statistics the model saw during training
      5. Reorder from height-width-channel to channel-height-width

    Returns a float32 array of shape (3, img_size, img_size).
    """
    w, h = pil_image.size
    short_target = int(round(img_size * resize_ratio))
    scale = short_target / float(min(w, h))
    new_w, new_h = max(img_size, int(round(w * scale))), max(img_size, int(round(h * scale)))
    resized = pil_image.resize((new_w, new_h), Image.BICUBIC)

    left = (new_w - img_size) // 2
    top = (new_h - img_size) // 2
    cropped = resized.crop((left, top, left + img_size, top + img_size))

    arr = np.asarray(cropped, dtype=np.float32) / 255.0 # (H, W, 3) in 0..1
    arr = (arr - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(arr.transpose(2, 0, 1)) # (3, H, W)


def load_and_preprocess(path, backbone):
    """Convenience wrapper: file path -> ready-to-embed array for `backbone`."""
    return preprocess(load_image(path), backbone.img_size, backbone.mean, backbone.std)


def embed_paths(paths, backbone, batch_size=4):
    """Embed a list of image files, a few at a time.

    Small batches keep peak memory low, which matters on a 4 GB board where
    the CPU and GPU share the same RAM.
    """
    chunks = []
    for start in range(0, len(paths), batch_size):
        batch = np.stack([load_and_preprocess(p, backbone)
                          for p in paths[start:start + batch_size]])
        chunks.append(backbone.embed(batch))
    return np.concatenate(chunks, axis=0)
