#!/usr/bin/env python3
"""
check_env.py - jetson type
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def line(label, value):
    print("%-28s %s" % (label + ":", value))


def main():
    try:
        from fewshot import profiler
    except ImportError as exc:
        print("Missing a basic dependency (%s). Install numpy, pillow and psutil first." % exc)
        return 1

    info = profiler.system_info()
    line("Board", info.get("jetson_model") or "not a Jetson (that is fine for development)")
    line("JetPack / L4T", info.get("l4t_release", "n/a"))
    line("CPU architecture", info.get("machine"))
    line("Python", info.get("python_version"))
    line("RAM total", "%s MB" % info.get("ram_total_mb", "?"))
    line("Swap total", "%s MB" % info.get("swap_total_mb", "?"))

    torch_ok = "torch_version" in info
    torch_gpu = bool(info.get("cuda_available"))
    line("PyTorch", info.get("torch_version", "not installed"))
    if torch_ok:
        line("  sees the GPU (CUDA)", "yes - %s" % info.get("cuda_device") if torch_gpu else "NO")
    line("timm", info.get("timm_version", "not installed"))

    ort_ok = "onnxruntime_version" in info
    providers = info.get("onnxruntime_providers", [])
    ort_gpu = "CUDAExecutionProvider" in providers
    line("ONNX Runtime", info.get("onnxruntime_version", "not installed"))
    if ort_ok:
        line("  providers", ", ".join(p.replace("ExecutionProvider", "") for p in providers))

    print("")
    print("Recommendation")
    print("--------------")
    py = sys.version_info
    if torch_ok and "timm_version" in info:
        print("* PyTorch route is available:  --backbone dinov2_vits14")
        if info.get("jetson_model") and not torch_gpu:
            print("  WARNING: PyTorch cannot see the GPU, so it would run on the CPU.")
            print("  You probably installed torch from PyPI; install NVIDIA's Jetson wheel")
            print("  instead (docs/JETSON_SETUP.md).")
    elif py < (3, 8):
        print("* Python %d.%d is too old for timm/DINOv2 in PyTorch (original Jetson Nano)." % (py[0], py[1]))
        print("  Use the ONNX route: export on another computer, then")
        print("  --backbone onnx:models/dinov2_vits14.onnx")
    else:
        print("* PyTorch route not ready: install torch (Jetson wheel) and timm.")
    if ort_ok:
        print("* ONNX route is available%s." % (
            " with GPU acceleration" if ort_gpu else " but CPU ONLY (no CUDA provider found)"))
    else:
        print("* ONNX route not ready: onnxruntime is not installed.")
    if info.get("swap_total_mb", 0) < 2000 and info.get("ram_total_mb", 99999) < 6000:
        print("* Less than 2 GB of swap on a small-memory board: add a swap file")
        print("  (docs/JETSON_SETUP.md, 'Add swap') to avoid out-of-memory crashes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
