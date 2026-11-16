#!/usr/bin/env python3
"""
predict.py - identify objects in new images
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fewshot import cli, data, profiler # noqa: E402
from fewshot.classifier import FewShotClassifier # noqa: E402


def classify_pil(pil_image, backbone, clf):
    """Identify one PIL image. Returns (prediction dict, latency in ms)."""
    t0 = profiler.now()
    x = data.preprocess(pil_image, backbone.img_size, backbone.mean, backbone.std)
    embedding = backbone.embed(x[np.newaxis])
    prediction = clf.predict(embedding)[0]
    return prediction, (profiler.now() - t0) * 1000.0


def csi_pipeline(width=1280, height=720, fps=30):
    """GStreamer pipeline string for a CSI camera on a Jetson.

    CSI cameras are not plain /dev/video devices, OpenCV has to read them
    through NVIDIA's `nvarguscamerasrc` GStreamer element.
    """
    return (
        "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%d, height=%d, "
        "framerate=%d/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! "
        "video/x-raw, format=BGR ! appsink drop=1" % (width, height, fps)
    )


def run_camera(source, backbone, clf, show):
    """Continuously classify frames from a camera until Ctrl+C (or 'q' in the window)."""
    import cv2
    from PIL import Image

    if source == "csi":
        cap = cv2.VideoCapture(csi_pipeline(), cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(int(source))
    if not cap.isOpened():
        raise RuntimeError("Could not open camera '%s'" % source)

    print("Camera running. Press Ctrl+C to stop.")
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            pred, ms = classify_pil(pil, backbone, clf)
            text = "%s  (score %.2f, %.0f ms)" % (pred["label"], pred["score"], ms)
            print(text)
            if show:
                cv2.putText(frame_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)
                cv2.imshow("fewshot-jetson", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--enrollment", default="enrollment.npz",
                        help="File produced by scripts/enroll.py")
    parser.add_argument("--image", help="An image file, or a folder of images.")
    parser.add_argument("--camera", help="Camera index (0, 1, ...) or 'csi'.")
    parser.add_argument("--show", action="store_true",
                        help="Camera mode: display a window (needs a monitor).")
    parser.add_argument("--json", action="store_true",
                        help="Print one JSON object per image instead of plain text.")
    cli.add_backbone_args(parser)
    args = parser.parse_args(argv)
    if cli.handle_list_backbones(args):
        return 0
    if not args.image and args.camera is None:
        parser.error("give --image <file or folder> or --camera <index|csi>")

    clf = FewShotClassifier.load(args.enrollment)
    backbone = cli.build_backbone(args)

    # Safety check
    if clf.backbone_spec and clf.backbone_spec != backbone.spec:
        sys.exit("ERROR: the enrolment was made with '%s' but you are predicting with "
                 "'%s'. Re-run enroll.py with this backbone, or pass the matching "
                 "--backbone." % (clf.backbone_spec, backbone.spec))
    if backbone.embed_dim != clf.prototypes.shape[1]:
        sys.exit("ERROR: embedding size mismatch between enrolment and backbone.")

    if args.camera is not None:
        run_camera(args.camera, backbone, clf, args.show)
        return 0


    if os.path.isdir(args.image):
        paths = []
        for root, _dirs, files in os.walk(args.image):
            paths.extend(os.path.join(root, f) for f in sorted(files)
                         if f.lower().endswith(data.IMAGE_EXTENSIONS))
        paths.sort()
    else:
        paths = [args.image]
    if not paths:
        sys.exit("No images found at %s" % args.image)

    for path in paths:
        pred, ms = classify_pil(data.load_image(path), backbone, clf)
        if args.json:
            print(json.dumps(dict(pred, image=path, latency_ms=round(ms, 2))))
        else:
            print("%-40s -> %-14s score %.3f  margin %.3f  %.1f ms" % (
                path, pred["label"], pred["score"], pred["margin"], ms))
    return 0


if __name__ == "__main__":
    sys.exit(main())
