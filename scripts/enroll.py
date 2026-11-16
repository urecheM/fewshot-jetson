#!/usr/bin/env python3
"""
enroll.py - teach objects to system
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fewshot import cli, data # noqa: E402
from fewshot.classifier import FewShotClassifier # noqa: E402


def enroll(backbone, support_dir, shots, seed, method="prototype", k=3,
           unknown_threshold=None, batch_size=4):
    """Build a fitted FewShotClassifier from a folder of labelled examples.

    Returned alongside the classifier is the dict of image paths actually used
    as shots, so callers (the benchmark) can record exactly what was enrolled.
    """
    all_paths = data.list_images_by_class(support_dir)
    chosen = data.select_shots(all_paths, shots, seed)
    class_names = sorted(chosen.keys())

    paths, labels = [], []
    for index, name in enumerate(class_names):
        paths.extend(chosen[name])
        labels.extend([index] * len(chosen[name]))

    embeddings = data.embed_paths(paths, backbone, batch_size=batch_size)

    clf = FewShotClassifier(method=method, k=k, unknown_threshold=unknown_threshold)
    clf.fit(embeddings, labels, class_names, backbone_spec=backbone.spec)
    return clf, chosen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--support", default="data/support",
                        help="Folder with one sub-folder of example images per object.")
    parser.add_argument("--shots", type=int, default=5,
                        help="Example images to use per class (0 = use all found).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed controlling WHICH images are picked as shots.")
    parser.add_argument("--method", choices=["prototype", "knn"], default="prototype")
    parser.add_argument("--k", type=int, default=3, help="Neighbours for --method knn.")
    parser.add_argument("--unknown-threshold", type=float, default=None,
                        help="Answer 'unknown' when the best similarity is below this "
                             "(e.g. 0.4). Default: always pick the closest class.")
    parser.add_argument("--output", default="enrollment.npz")
    cli.add_backbone_args(parser)
    args = parser.parse_args(argv)
    if cli.handle_list_backbones(args):
        return 0

    print("Loading backbone '%s' ..." % args.backbone)
    backbone = cli.build_backbone(args)
    print("  running on %s (%s), embedding size %d, loaded in %.1f s" % (
        backbone.device, backbone.precision, backbone.embed_dim, backbone.load_time_s))

    clf, chosen = enroll(backbone, args.support, args.shots, args.seed,
                         method=args.method, k=args.k,
                         unknown_threshold=args.unknown_threshold)
    for name in clf.class_names:
        print("  enrolled %-15s from %d image(s)" % (name, len(chosen[name])))

    clf.save(args.output)
    print("Saved enrolment to %s" % args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
