"""
profiler.py - measuring computational performance (requirement c).
"""

import os
import platform
import sys
import threading
import time

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

MB = 1024.0 * 1024.0


# Timing
def now():
    """Current time in seconds from a monotonic high-resolution clock."""
    return time.perf_counter()


def summarize_ms(samples_ms):
    """Turn a list of per-image timings (milliseconds) into labelled statistics."""
    if len(samples_ms) == 0:
        return {"count": 0}
    a = np.asarray(samples_ms, dtype=np.float64)
    return {
        "count": int(a.size),
        "mean_ms": round(float(a.mean()), 3),
        "median_ms": round(float(np.median(a)), 3),
        "std_ms": round(float(a.std()), 3),
        "min_ms": round(float(a.min()), 3),
        "max_ms": round(float(a.max()), 3),
        "p90_ms": round(float(np.percentile(a, 90)), 3),
        "p95_ms": round(float(np.percentile(a, 95)), 3),
        "p99_ms": round(float(np.percentile(a, 99)), 3),
    }


# Memory
def _read_proc_meminfo():
    """Fallback for system memory when psutil is missing (Linux only)."""
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, value = line.split(":", 1)
                info[key] = float(value.strip().split()[0]) * 1024.0
    except (IOError, OSError, ValueError):
        pass
    return info


def memory_snapshot():
    """One instantaneous reading of process and system memory, in MB."""
    snap = {}
    if psutil is not None:
        snap["process_rss_mb"] = psutil.Process(os.getpid()).memory_info().rss / MB
        vm = psutil.virtual_memory()
        snap["system_total_mb"] = vm.total / MB
        # used = total - available
        snap["system_used_mb"] = (vm.total - vm.available) / MB
        snap["swap_used_mb"] = psutil.swap_memory().used / MB
    else:
        info = _read_proc_meminfo()
        if info:
            snap["system_total_mb"] = info.get("MemTotal", 0) / MB
            snap["system_used_mb"] = (info.get("MemTotal", 0) - info.get("MemAvailable", 0)) / MB
            snap["swap_used_mb"] = (info.get("SwapTotal", 0) - info.get("SwapFree", 0)) / MB
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        snap["process_rss_mb"] = float(line.split()[1]) / 1024.0
        except (IOError, OSError):
            pass
    return snap


class MemorySampler(object):
    """Background thread that records the PEAK memory use during a run.

    Usage:
        sampler = MemorySampler(); sampler.start()
        ... work ...
        sampler.mark("after_model_load") # named checkpoint
        ... more work ...
        report = sampler.stop()
    """

    def __init__(self, interval_s=0.05):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = None
        self.baseline = {}
        self.peak = {}
        self.marks = {}
        self.num_samples = 0

    def _update_peak(self, snap):
        for key, value in snap.items():
            if key == "system_total_mb":
                continue
            if value > self.peak.get(key, float("-inf")):
                self.peak[key] = value

    def _run(self):
        while not self._stop.is_set():
            self._update_peak(memory_snapshot())
            self.num_samples += 1
            self._stop.wait(self.interval_s)

    def start(self):
        self.baseline = memory_snapshot()
        self._update_peak(self.baseline)
        self._thread = threading.Thread(target=self._run, name="memory-sampler")
        self._thread.daemon = True
        self._thread.start()
        return self

    def mark(self, name):
        """Record memory at a named point in the run (e.g. 'after_model_load')."""
        snap = memory_snapshot()
        self._update_peak(snap)
        self.marks[name] = snap
        return snap

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._update_peak(memory_snapshot())
        return self.report()

    def report(self):
        """Labelled summary."""
        def r(value):
            return None if value is None else round(value, 1)

        base, peak = self.baseline, self.peak
        after_load = self.marks.get("after_model_load", {})
        out = {
            "system_total_mb": r(base.get("system_total_mb")),
            # this python process
            "process_rss_baseline_mb": r(base.get("process_rss_mb")),
            "process_rss_after_model_load_mb": r(after_load.get("process_rss_mb")),
            "process_rss_peak_mb": r(peak.get("process_rss_mb")),
            # whole board
            "system_used_baseline_mb": r(base.get("system_used_mb")),
            "system_used_after_model_load_mb": r(after_load.get("system_used_mb")),
            "system_used_peak_mb": r(peak.get("system_used_mb")),
            "swap_used_peak_mb": r(peak.get("swap_used_mb")),
            "sampling_interval_ms": int(self.interval_s * 1000),
            "num_samples": self.num_samples,
        }
        if base.get("process_rss_mb") is not None and peak.get("process_rss_mb") is not None:
            out["process_rss_increase_peak_mb"] = r(peak["process_rss_mb"] - base["process_rss_mb"])
        if base.get("system_used_mb") is not None and peak.get("system_used_mb") is not None:
            out["system_used_increase_peak_mb"] = r(peak["system_used_mb"] - base["system_used_mb"])
        return out


# Jetson-specific readings
def jetson_model():
    """Return board name or None.

    Every Jetson shows model name in the device tree.
    """
    try:
        with open("/proc/device-tree/model") as f:
            return f.read().strip("\x00\n ")
    except (IOError, OSError):
        return None


class JetsonStatsSampler(object):
    """Collect power, GPU load and temperature through `jetson-stats` (jtop).
    """

    def __init__(self):
        self.available = False
        self.note = ""
        self._rows = []
        self._stop = threading.Event()
        self._thread = None
        try:
            from jtop import jtop # noqa: F401
            self.available = True
        except Exception as exc:   # ImportError, or jtop service not running
            self.note = "jetson-stats not available (%s)" % exc.__class__.__name__

    def _run(self):
        from jtop import jtop
        try:
            with jtop() as jetson:
                # jetson.ok() blocks until the next reading is ready (~1 s) + returns False when the service goes away
                while jetson.ok() and not self._stop.is_set():
                    self._rows.append(dict(jetson.stats))
        except Exception as exc:
            self.note = "jtop failed: %s" % exc

    def start(self):
        if self.available:
            self._thread = threading.Thread(target=self._run, name="jtop-sampler")
            self._thread.daemon = True
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return self.report()

    def report(self):
        if not self._rows:
            return {"available": False,
                    "note": self.note or "no samples collected (run was shorter than ~1 s?)"}
        summary = {"available": True, "num_samples": len(self._rows), "readings": {}}
        keys = sorted({k for row in self._rows for k in row})
        for key in keys:
            lower = key.lower()
            if not any(word in lower for word in ("power", "gpu", "temp")):
                continue
            values = [row[key] for row in self._rows
                      if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)]
            if values:
                summary["readings"][key] = {
                    "mean": round(float(np.mean(values)), 2),
                    "max": round(float(np.max(values)), 2),
                }
        summary["units_note"] = ("jtop units: power in milliwatts, GPU in percent load, "
                                 "temperatures in degrees Celsius.")
        return summary

def system_info():
    info = {
        "jetson_model": jetson_model(),
        "machine": platform.machine(), # aarch64 on a Jetson
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "numpy_version": np.__version__,
    }
    if psutil is not None:
        info["ram_total_mb"] = round(psutil.virtual_memory().total / MB, 1)
        info["swap_total_mb"] = round(psutil.swap_memory().total / MB, 1)
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        import timm
        info["timm_version"] = timm.__version__
    except ImportError:
        pass
    try:
        import onnxruntime
        info["onnxruntime_version"] = onnxruntime.__version__
        info["onnxruntime_providers"] = onnxruntime.get_available_providers()
    except ImportError:
        pass
    try:
        with open("/etc/nv_tegra_release") as f:
            info["l4t_release"] = f.readline().strip()
    except (IOError, OSError):
        pass
    return info
