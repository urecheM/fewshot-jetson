#!/usr/bin/env python3
"""
export_onnx.py - backbone to onnx conversion
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fewshot import cli # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="Destination .onnx path.")
    parser.add_argument("--opset", type=int, default=14,
                        help="ONNX operator-set version. 14 is new enough for ViT "
                             "attention and old enough for the ONNX Runtime 1.10/1.11 "
                             "builds available for JetPack 4.6.")
    cli.add_backbone_args(parser)
    args = parser.parse_args(argv)
    if cli.handle_list_backbones(args):
        return 0

    import torch

    try:
        import timm.layers
        timm.layers.set_fused_attn(False)
    except Exception:
        pass

    args.device, args.fp16 = "cpu", False
    backbone = cli.build_backbone(args)
    if backbone.backend == "onnx":
        sys.exit("The backbone is already an ONNX file; nothing to export.")
    model = backbone._model

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    dummy = torch.zeros(1, 3, backbone.img_size, backbone.img_size)
    export_kwargs = dict(
        input_names=["image"],
        output_names=["embedding"],
        opset_version=args.opset,
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        do_constant_folding=True,
    )
    print("Exporting %s to %s (opset %d) ..." % (backbone.spec, args.output, args.opset))
    try:
        torch.onnx.export(model, dummy, args.output, dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, args.output, **export_kwargs)

    meta = {
        "source_spec": backbone.spec,
        "img_size": backbone.img_size,
        "mean": list(backbone.mean),
        "std": list(backbone.std),
        "embed_dim": backbone.embed_dim,
        "num_parameters": backbone.num_parameters,
        "precision": "fp32",
        "opset": args.opset,
        "pretrained": not args.no_pretrained,
    }
    with open(args.output + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Wrote %s (%.1f MB) and %s.json" % (
        args.output, os.path.getsize(args.output) / 1e6, args.output))

    try:
        from fewshot.backbones import OnnxBackbone
        onnx_bb = OnnxBackbone(args.output, providers=["CPUExecutionProvider"])
    except ImportError:
        print("onnxruntime not installed here - skipping the verification step.")
        return 0
    rng = np.random.RandomState(0)
    x = rng.randn(2, 3, backbone.img_size, backbone.img_size).astype(np.float32)
    ref, got = backbone.embed(x), onnx_bb.embed(x)
    cos = float(np.sum(ref * got, axis=1).min())
    print("Verification: minimum cosine similarity PyTorch vs ONNX = %.6f" % cos)
    if cos < 0.999:
        sys.exit("ERROR: exported model does not match the original. Do not use it.")
    print("OK - exported model matches.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
