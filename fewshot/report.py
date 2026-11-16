"""
report.py - writing benchmark results
"""

import csv
import json
import os


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def write_csv(path, rows, fieldnames=None):
    """Write a list of dicts as CSV. Column order follows `fieldnames`."""
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(value, digits=1):
    """Format a number for a Markdown table; show a dash for missing values."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return ("%." + str(digits) + "f") % value
    return str(value)


def _latency_row(label, stats):
    return "| %s | %s | %s | %s | %s | %s | %s | %s |" % (
        label, _fmt(stats.get("mean_ms"), 2), _fmt(stats.get("median_ms"), 2),
        _fmt(stats.get("p95_ms"), 2), _fmt(stats.get("p99_ms"), 2),
        _fmt(stats.get("min_ms"), 2), _fmt(stats.get("max_ms"), 2),
        _fmt(stats.get("std_ms"), 2))


def render_summary_markdown(m):
    """Build the human-readable report from the metrics dictionary `m`."""
    run, cfg, model, sysinfo = m["run"], m["config"], m["model"], m["system"]
    lat, thr, mem, acc = m["latency_ms"], m["throughput"], m["memory_mb"], m["accuracy"]
    lines = []
    add = lines.append

    add("# Benchmark run: %s" % run["name"])
    add("")
    add("- **Date (UTC):** %s" % run["timestamp_utc"])
    add("- **Device:** %s (%s)" % (sysinfo.get("jetson_model") or "not a Jetson", sysinfo.get("machine")))
    add("- **Model:** `%s`" % model["spec"])
    add("- **Runs on:** %s, %s, %d x %d px input" % (
        model["device"], model["precision"], model["img_size_px"], model["img_size_px"]))
    add("- **Few-shot setup:** %s shots per class, `%s` classifier, %d classes (%s)" % (
        cfg["shots_per_class_used"], cfg["method"], len(m["dataset"]["classes"]),
        ", ".join(m["dataset"]["classes"])))
    if not cfg.get("pretrained", True):
        add("- **WARNING:** run with random (untrained) weights - accuracy is meaningless, "
            "timings and memory are still valid.")
    add("")

    add("## Headline numbers")
    add("")
    add("| Metric | Value |")
    add("|---|---|")
    add("| Accuracy (top-1) | %s %% (%d / %d images) |" % (
        _fmt(acc["top1_accuracy_percent"]), acc["num_correct"], acc["num_evaluated"]))
    add("| Latency per image, end-to-end (median) | %s ms |" % _fmt(lat["end_to_end"].get("median_ms"), 2))
    add("| Latency per image, model inference only (median) | %s ms |" % _fmt(lat["inference"].get("median_ms"), 2))
    add("| Throughput, one image at a time, end-to-end | %s images/s |" % _fmt(thr["end_to_end_images_per_s"], 2))
    if thr.get("batched", {}).get("images_per_s") is not None:
        add("| Throughput, batched inference (batch = %d) | %s images/s |" % (
            thr["batched"]["batch_size"], _fmt(thr["batched"]["images_per_s"], 2)))
    add("| Peak process memory (RSS) | %s MB |" % _fmt(mem.get("process_rss_peak_mb")))
    add("| Peak increase in system memory | %s MB |" % _fmt(mem.get("system_used_increase_peak_mb")))
    if mem.get("gpu_torch_peak_reserved_mb") is not None:
        add("| Peak GPU memory reserved by PyTorch | %s MB |" % _fmt(mem.get("gpu_torch_peak_reserved_mb")))
    add("| Model load time | %s s |" % _fmt(model["load_time_s"], 2))
    add("| Model size | %s M parameters, %d-dim embedding |" % (
        _fmt(model.get("num_parameters_millions"), 2), model["embed_dim"]))
    add("")

    add("## Latency (milliseconds per image, batch size 1)")
    add("")
    add("Measured over %d timed images (%d query images x %d repeats) after %d warm-up inferences." % (
        lat["end_to_end"].get("count", 0), m["dataset"]["num_query_images"],
        cfg["repeats"], cfg["warmup"]))
    add("")
    add("| Stage | mean | median | p95 | p99 | min | max | std |")
    add("|---|---|---|---|---|---|---|---|")
    add(_latency_row("Load + preprocess image", lat["load_and_preprocess"]))
    add(_latency_row("Model inference (embedding)", lat["inference"]))
    add(_latency_row("Classification", lat["classify"]))
    add(_latency_row("**End-to-end**", lat["end_to_end"]))
    add("")

    add("## Throughput (images per second)")
    add("")
    add("| Mode | images/s |")
    add("|---|---|")
    add("| End-to-end, one image at a time | %s |" % _fmt(thr["end_to_end_images_per_s"], 2))
    add("| Inference only, one image at a time | %s |" % _fmt(thr["inference_only_images_per_s"], 2))
    b = thr.get("batched", {})
    if b.get("images_per_s") is not None:
        add("| Inference only, batch of %d | %s |" % (b["batch_size"], _fmt(b["images_per_s"], 2)))
    elif b.get("error"):
        add("| Inference only, batch of %s | failed: %s |" % (b.get("batch_size"), b["error"]))
    add("")

    add("## Memory (megabytes)")
    add("")
    add("| Measurement | Baseline | After model load | Peak |")
    add("|---|---|---|---|")
    add("| This process (RSS) | %s | %s | %s |" % (
        _fmt(mem.get("process_rss_baseline_mb")), _fmt(mem.get("process_rss_after_model_load_mb")),
        _fmt(mem.get("process_rss_peak_mb"))))
    add("| Whole system in use | %s | %s | %s |" % (
        _fmt(mem.get("system_used_baseline_mb")), _fmt(mem.get("system_used_after_model_load_mb")),
        _fmt(mem.get("system_used_peak_mb"))))
    add("")
    add("- System RAM installed: %s MB" % _fmt(mem.get("system_total_mb")))
    add("- Peak swap in use: %s MB (non-zero growth means RAM ran out and timings suffered)" % _fmt(mem.get("swap_used_peak_mb")))
    if mem.get("gpu_torch_peak_allocated_mb") is not None:
        add("- PyTorch GPU memory, peak allocated / reserved: %s / %s MB" % (
            _fmt(mem.get("gpu_torch_peak_allocated_mb")), _fmt(mem.get("gpu_torch_peak_reserved_mb"))))
    add("")

    add("## Accuracy")
    add("")
    add("| Class | Correct | Total | Accuracy % |")
    add("|---|---|---|---|")
    for name, row in acc["per_class"].items():
        add("| %s | %d | %d | %s |" % (name, row["correct"], row["total"], _fmt(row["accuracy_percent"])))
    add("")
    cm = acc["confusion_matrix"]
    add("Confusion matrix (rows = true class, columns = predicted class):")
    add("")
    add("| true \\ predicted | " + " | ".join(cm["predicted_labels"]) + " |")
    add("|---|" + "---|" * len(cm["predicted_labels"]))
    for true_name, counts in zip(cm["true_labels"], cm["counts"]):
        add("| **%s** | " % true_name + " | ".join(str(c) for c in counts) + " |")
    add("")

    jet = m.get("jetson_stats", {})
    if jet.get("available"):
        add("## Jetson power / GPU / temperature (from jetson-stats)")
        add("")
        add("| Reading | mean | max |")
        add("|---|---|---|")
        for key, v in jet["readings"].items():
            add("| %s | %s | %s |" % (key, _fmt(v["mean"], 2), _fmt(v["max"], 2)))
        add("")
        add(jet.get("units_note", ""))
        add("")

    add("## Glossary")
    add("")
    add("- **Latency**: time to process one image. *Median* is the typical case; *p95/p99* "
        "are the slow cases (95 % / 99 % of images finished within this time).")
    add("- **End-to-end**: reading the file from disk, preprocessing, running the model and classifying. "
        "*Inference only* is just the neural network.")
    add("- **Throughput**: images processed per second. Batching feeds several images to the model "
        "at once, which is usually more efficient but needs more memory.")
    add("- **Warm-up**: the first few inferences are slower (GPU initialisation, caches) and are "
        "run but not timed.")
    add("- **RSS** (resident set size): physical RAM held by this Python process.")
    add("- **System memory in use**: total RAM minus RAM still available, for the whole board. "
        "On a Jetson the GPU shares this RAM, so this captures GPU usage as well.")
    add("- **Peak increase in system memory**: peak minus baseline - the best single estimate of the RAM "
        "this run needed, provided nothing else heavy was running.")
    add("- **Shots**: number of labelled example images per class.")
    add("")
    return "\n".join(lines)


def write_run_outputs(out_dir, metrics, per_image_rows):
    """Write all three report files for a run and return their paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "metrics_json": os.path.join(out_dir, "metrics.json"),
        "summary_md": os.path.join(out_dir, "summary.md"),
        "per_image_csv": os.path.join(out_dir, "per_image.csv"),
    }
    write_json(paths["metrics_json"], metrics)
    with open(paths["summary_md"], "w") as f:
        f.write(render_summary_markdown(metrics))
    write_csv(paths["per_image_csv"], per_image_rows)
    return paths
