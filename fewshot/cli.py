"""
cli.py - command-line options shared by all scripts
"""

from .backbones import create_backbone, format_presets


def add_backbone_args(parser):
    """Attach the model-selection options to an argparse parser."""
    g = parser.add_argument_group("model selection")
    g.add_argument("--backbone", default="dinov2_vits14",
                   help="Preset name (see --list-backbones) or a full spec: "
                        "timm:<model>, hub:<user>/<repo>:<entry>, onnx:<file.onnx>. "
                        "Default: dinov2_vits14")
    g.add_argument("--device", default="auto",
                   help="auto (GPU if PyTorch can see one), cuda, or cpu. "
                        "Ignored by the onnx backend (see --ort-providers).")
    g.add_argument("--fp16", action="store_true",
                   help="Half-precision inference on the GPU: about half the model "
                        "memory and usually faster. PyTorch backends only.")
    g.add_argument("--img-size", type=int, default=224,
                   help="Square input size in pixels. For DINOv2 (patch 14) use a "
                        "multiple of 14, e.g. 224. Smaller = faster, less accurate.")
    g.add_argument("--no-pretrained", action="store_true",
                   help="Use random weights instead of downloading pretrained ones. "
                        "ONLY for checking the pipeline runs (accuracy will be chance).")
    g.add_argument("--ort-providers", default=None,
                   help="Comma-separated ONNX Runtime providers, in priority order, "
                        "e.g. CUDAExecutionProvider,CPUExecutionProvider")
    g.add_argument("--list-backbones", action="store_true",
                   help="Print the preset model names and exit.")


def handle_list_backbones(args):
    """If --list-backbones was given, print the table and tell the caller to exit."""
    if getattr(args, "list_backbones", False):
        print(format_presets())
        return True
    return False


def build_backbone(args):
    """Create the backbone described by the parsed command-line arguments."""
    providers = args.ort_providers.split(",") if args.ort_providers else None
    return create_backbone(
        args.backbone,
        device=args.device,
        fp16=args.fp16,
        img_size=args.img_size,
        pretrained=not args.no_pretrained,
        ort_providers=providers,
    )
