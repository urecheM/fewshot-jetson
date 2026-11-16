#!/usr/bin/env python3
"""
compare_runs.py - side by side comparison for several benchmark runs
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fewshot import report # noqa: E402

COLUMNS = [
    ("run", lambda m: m["run"]["name"]),
    ("model", lambda m: m["config"]["backbone_argument"]),
    ("device", lambda m: m["model"]["device"]),
    ("precision", lambda m: m["model"]["precision"]),
    ("shots", lambda m: m["config"]["shots_per_class_used"]),
    ("method", lambda m: m["config"]["method"]),
    ("accuracy_percent", lambda m: m["accuracy"]["top1_accuracy_percent"]),
    ("latency_e2e_median_ms", lambda m: m["latency_ms"]["end_to_end"].get("median_ms")),
    ("latency_e2e_p95_ms", lambda m: m["latency_ms"]["end_to_end"].get("p95_ms")),
    ("latency_inference_median_ms", lambda m: m["latency_ms"]["inference"].get("median_ms")),
    ("throughput_e2e_images_per_s", lambda m: m["throughput"]["end_to_end_images_per_s"]),
    ("throughput_batched_images_per_s", lambda m: m["throughput"].get("batched", {}).get("images_per_s")),
    ("process_rss_peak_mb", lambda m: m["memory_mb"].get("process_rss_peak_mb")),
    ("system_mem_increase_peak_mb", lambda m: m["memory_mb"].get("system_used_increase_peak_mb")),
    ("gpu_torch_peak_reserved_mb", lambda m: m["memory_mb"].get("gpu_torch_peak_reserved_mb")),
    ("model_load_time_s", lambda m: m["model"]["load_time_s"]),
    ("parameters_millions", lambda m: m["model"].get("num_parameters_millions")),
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results", help="Folder containing run folders.")
    args = parser.parse_args(argv)

    files = sorted(glob.glob(os.path.join(args.results, "*", "metrics.json")))
    if not files:
        sys.exit("No runs found under %s - run scripts/benchmark.py first." % args.results)

    rows = []
    for path in files:
        with open(path) as f:
            m = json.load(f)
        row = {}
        for title, getter in COLUMNS:
            try:
                row[title] = getter(m)
            except (KeyError, TypeError):
                row[title] = None # tolerate reports from older versions
        rows.append(row)

    titles = [t for t, _ in COLUMNS]
    csv_path = os.path.join(args.results, "comparison.csv")
    report.write_csv(csv_path, rows, fieldnames=titles)

    md_path = os.path.join(args.results, "comparison.md")
    with open(md_path, "w") as f:
        f.write("# Run comparison\n\n")
        f.write("| " + " | ".join(titles) + " |\n")
        f.write("|" + "---|" * len(titles) + "\n")
        for row in rows:
            f.write("| " + " | ".join("-" if row[t] is None else str(row[t]) for t in titles) + " |\n")
    print("Wrote %s\nWrote %s" % (csv_path, md_path))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed - skipping comparison.png)")
        return 0

    charts = [
        ("accuracy_percent", "Top-1 accuracy (%) - higher is better"),
        ("latency_e2e_median_ms", "Median end-to-end latency (ms) - lower is better"),
        ("throughput_e2e_images_per_s", "Throughput (images/s) - higher is better"),
        ("process_rss_peak_mb", "Peak process memory (MB) - lower is better"),
    ]
    labels = [r["run"] for r in rows]
    fig, axes = plt.subplots(len(charts), 1, figsize=(9, 2.2 * len(charts) + 0.4 * len(rows)))
    for ax, (key, title) in zip(axes, charts):
        values = [r[key] or 0 for r in rows]
        bars = ax.barh(labels, values, color="#3b6ea5")
        ax.set_title(title, loc="left", fontsize=10)
        ax.invert_yaxis() #first run at top
        for bar, value in zip(bars, values):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                    " %.1f" % value, va="center", fontsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    png_path = os.path.join(args.results, "comparison.png")
    fig.savefig(png_path, dpi=130)
    print("Wrote %s" % png_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
