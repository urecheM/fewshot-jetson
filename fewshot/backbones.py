import json
import os
import time

import numpy as np

# ImageNet channel statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


PRESETS = {
    # DINOv2 (Meta AI, Apache-2.0). "s"=small (22M parameters), "b"=base (87M)
    # default
    "dinov2_vits14": "timm:vit_small_patch14_dinov2.lvd142m",
    "dinov2_vitb14": "timm:vit_base_patch14_dinov2.lvd142m",
    # DINOv2 "with registers"
    "dinov2_vits14_reg": "timm:vit_small_patch14_reg4_dinov2.lvd142m",
    # The same DINOv2 ViT-S/14 loaded thru Meta's repo instead of timm
    "dinov2_vits14_hub": "hub:facebookresearch/dinov2:dinov2_vits14",
    # DINO v1: predecessor of DINOv2 (self-supervised, patch size 16)
    "dino_vits16": "timm:vit_small_patch16_224.dino",
    # DINOv3: the successor of v2
    "dinov3_vits16": "timm:vit_small_patch16_dinov3.lvd1689m",
    # CLIP image tower: trained on image-text pairs rather than images alone
    "clip_vitb32": "timm:vit_base_patch32_clip_224.openai",
    # DeiT: a ViT of the same size as v2-S but trained WITH ImageNet labels
    "deit_small": "timm:deit_small_patch16_224.fb_in1k",
    # Classic CNN baselines (cheaper to run than a ViT)
    "resnet50": "timm:resnet50.a1_in1k",
    "resnet18": "timm:resnet18.a1_in1k",
    "efficientnet_b0": "timm:efficientnet_b0.ra_in1k",
    "mobilenetv3_large": "timm:mobilenetv3_large_100.ra_in1k",
}


def l2_normalize(x, eps=1e-12):

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


class Backbone(object):
    """backbone"""

    spec = ""
    backend = ""
    embed_dim = 0
    img_size = 224
    mean = IMAGENET_MEAN
    std = IMAGENET_STD
    device = "cpu"
    precision = "fp32"
    load_time_s = 0.0
    num_parameters = None

    def embed(self, batch):
        """batch: float32 array (N, 3, H, W), already normalised -> (N, embed_dim)"""
        raise NotImplementedError

    def gpu_memory_mb(self):
        """Return GPU memory counters in MB, or {} if the backend cannot report them"""
        return {}

    def reset_gpu_peak(self):
        """Reset 'peak GPU memory' counters so a new measurement phase starts new"""
        pass

    def describe(self):
        """Params of this model in benchmark report"""
        return {
            "spec": self.spec,
            "backend": self.backend,
            "device": self.device,
            "precision": self.precision,
            "img_size_px": self.img_size,
            "embed_dim": self.embed_dim,
            "num_parameters_millions": (
                round(self.num_parameters / 1e6, 2) if self.num_parameters else None
            ),
            "load_time_s": round(self.load_time_s, 3),
        }


class _TorchBackbone(Backbone):
    """Shared logic for the two PyTorch-based backends (timm and torch.hub)"""

    def _finish_setup(self, model, device, fp16):
        # torch is imported inside methods (not at the top of the file) so that
        # the ONNX backend can be used on a machine with no PyTorch installed.
        import torch

        self._torch = torch

        # auto picks the GPU when PyTorch can see one
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # eval() switches off training-only behaviour such as dropout so that
        # the same image always gives the same embedding
        model.eval()
        model.to(device)

        # FP16 stores each number in 16 bits instead of 32 (half-precision)
        self._fp16 = bool(fp16) and device.startswith("cuda")
        if self._fp16:
            model.half()
        self.precision = "fp16" if self._fp16 else "fp32"

        self._model = model
        self.num_parameters = sum(p.numel() for p in model.parameters())

        # discover embedding length by pushing one blank image
        dummy = np.zeros((1, 3, self.img_size, self.img_size), dtype=np.float32)
        self.embed_dim = int(self.embed(dummy).shape[1])

    def embed(self, batch):
        torch = self._torch
        x = torch.from_numpy(np.ascontiguousarray(batch)).to(self.device)
        if self._fp16:
            x = x.half()
        # tells PyTorch not to record operations for back-propagation
        with torch.no_grad():
            feats = self._model(x)
        if feats.dim() == 4:
            feats = feats.mean(dim=(2, 3))
        elif feats.dim() == 3:
            feats = feats.mean(dim=1)
        out = feats.float().cpu().numpy()
        return l2_normalize(out).astype(np.float32)

    def gpu_memory_mb(self):
        torch = self._torch
        if not self.device.startswith("cuda"):
            return {}
        mb = 1024.0 * 1024.0
        return {
            # Memory held by tensors (weights+activations) (current)
            "gpu_torch_allocated_mb": round(torch.cuda.memory_allocated() / mb, 1),
            # highest value of the above since reset
            "gpu_torch_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / mb, 1),
            # Memory used including cache
            "gpu_torch_peak_reserved_mb": round(torch.cuda.max_memory_reserved() / mb, 1),
        }

    def reset_gpu_peak(self):
        if self.device.startswith("cuda"):
            self._torch.cuda.reset_peak_memory_stats()


class TimmBackbone(_TorchBackbone):
    """Any model from the `timm` library, used as a headless feature extractor"""

    backend = "timm"

    def __init__(self, model_name, device="auto", fp16=False, img_size=224, pretrained=True):
        import timm

        t0 = time.perf_counter()
        self.spec = "timm:" + model_name

        try:
            model = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0, img_size=img_size
            )
        except TypeError:
            model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.img_size = int(img_size)

        # Identify normalisation the model was trained with via timm
        cfg = timm.data.resolve_model_data_config(model)
        self.mean = tuple(float(v) for v in cfg.get("mean", IMAGENET_MEAN))
        self.std = tuple(float(v) for v in cfg.get("std", IMAGENET_STD))

        self._finish_setup(model, device, fp16)
        self.load_time_s = time.perf_counter() - t0


class HubBackbone(_TorchBackbone):
    """A model loaded with torch.hub

    Spec format:  hub:<github user>/<repo>:<entry point>
    The first use downloads the repo's code and weights to ~/.cache/torch
    """

    backend = "hub"

    def __init__(self, repo_and_entry, device="auto", fp16=False, img_size=224, pretrained=True):
        import torch

        t0 = time.perf_counter()
        self.spec = "hub:" + repo_and_entry
        if ":" not in repo_and_entry:
            raise ValueError(
                "hub backbones are written 'hub:<user>/<repo>:<entry>', "
                "for example hub:facebookresearch/dinov2:dinov2_vits14"
            )
        repo, entry = repo_and_entry.split(":", 1)
        model = torch.hub.load(repo, entry, pretrained=pretrained)
        self.img_size = int(img_size)
        self._finish_setup(model, device, fp16)
        self.load_time_s = time.perf_counter() - t0


class OnnxBackbone(Backbone):
    """A model exported to the ONNX format and run with ONNX Runtime.

    scripts/export_onnx.py writes `model.onnx` and `model.onnx.json`
    """

    backend = "onnx"

    def __init__(self, path, providers=None):
        import onnxruntime as ort

        t0 = time.perf_counter()
        self.spec = "onnx:" + os.path.basename(path)
        if not os.path.isfile(path):
            raise FileNotFoundError("ONNX model not found: %s" % path)

        # read sidecar metadata
        meta = {}
        meta_path = path + ".json"
        if os.path.isfile(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
        self.img_size = int(meta.get("img_size", 224))
        self.mean = tuple(meta.get("mean", IMAGENET_MEAN))
        self.std = tuple(meta.get("std", IMAGENET_STD))
        self.source_spec = meta.get("source_spec", "unknown")
        self.num_parameters = meta.get("num_parameters")

        # choose execution provider
        available = ort.get_available_providers()
        if not providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in providers if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name

        used = self._session.get_providers()
        self.device = "onnxruntime:" + used[0].replace("ExecutionProvider", "").lower()
        self.precision = meta.get("precision", "fp32")

        dummy = np.zeros((1, 3, self.img_size, self.img_size), dtype=np.float32)
        self.embed_dim = int(self.embed(dummy).shape[1])
        self.load_time_s = time.perf_counter() - t0

    def embed(self, batch):
        # session.run is synchronous and returns only when result is ready.
        out = self._session.run(None, {self._input_name: batch.astype(np.float32)})[0]
        return l2_normalize(out).astype(np.float32)

    def describe(self):
        info = Backbone.describe(self)
        info["exported_from"] = self.source_spec
        return info


def resolve_spec(name):
    """Translate a preset alias into a full 'backend:identifier' specification."""
    spec = PRESETS.get(name, name)
    if ":" not in spec:
        spec = "timm:" + spec
    return spec


def create_backbone(name, device="auto", fp16=False, img_size=224, pretrained=True,
                    ort_providers=None):
    """build the right Backbone subclass from a name or specification.

    The only function the scripts call.
    """
    spec = resolve_spec(name)
    backend, ident = spec.split(":", 1)
    if backend == "timm":
        return TimmBackbone(ident, device=device, fp16=fp16, img_size=img_size,
                            pretrained=pretrained)
    if backend == "hub":
        return HubBackbone(ident, device=device, fp16=fp16, img_size=img_size,
                           pretrained=pretrained)
    if backend == "onnx":
        return OnnxBackbone(ident, providers=ort_providers)
    raise ValueError("Unknown backend '%s' in backbone spec '%s' "
                     "(expected timm:, hub: or onnx:)" % (backend, spec))


def format_presets():
    """Human-readable table of the preset names, for --list-backbones."""
    width = max(len(k) for k in PRESETS)
    lines = ["%-*s  ->  %s" % (width, k, v) for k, v in PRESETS.items()]
    lines.append("")
    lines.append("Anything else is accepted too:  timm:<model>,  hub:<user>/<repo>:<entry>,  onnx:<file.onnx>")
    return "\n".join(lines)
