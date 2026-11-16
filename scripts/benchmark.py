#!/usr/bin/env python3
"""
benchmark.py - complete test run
"""

import argparse
import datetime
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fewshot import cli, data, profiler, report # noqa: E402
from fewshot.classifier import UNKNOWN_LABEL # noqa: E402
from scripts.enroll import enroll # noqa: E402


def measure_batched_throughput(backbone, tensors, batch_size, repeats):
    """Images per second when the model is fed `batch_size` images at once.
    """
    result = {"batch_size": batch_size, "images_per_s": None}
    try:
        batches = [np.stack(tensors[i:i + batch_size])
                   for i in range(0, len(tensors), batch_size)]
        backbone.embed(batches[0])
        images, batch_ms = 0, []
        t_start = profiler.now()
        for _ in range(repeats):
            for batch in batches:
                t0 = profiler.now()
                backbone.embed(batch)
                batch_ms.append((profiler.now() - t0) * 1000.0)
                images += len(batch)
        elapsed = profiler.now() - t_start
        result["images_per_s"] = round(images / elapsed, 3)
        result["num_images_timed"] = images
        result["latency_per_batch"] = profiler.summarize_ms(batch_ms)
    except Exception as exc: # ex) CUDA out of memory
        result["error"] = "%s: %s" % (exc.__class__.__name__, str(exc)[:200])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--support", default="data/support",
                        help="Folder of labelled example images (one sub-folder per class).")
    parser.add_argument("--query", default="data/query",
                        help="Folder of test images in the same layout; the sub-folder "
                             "name is the correct answer used to score accuracy.")
    parser.add_argument("--shots", type=int, default=5,
                        help="Labelled examples per class (0 = all). Try 1, 3, 5, 10.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", choices=["prototype", "knn"], default="prototype")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--unknown-threshold", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=10,
                        help="Untimed inferences before measuring. The first calls on a "
                             "GPU are much slower and would distort the statistics.")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Passes over the query set. More passes = steadier numbers.")
    parser.add_argument("--throughput-batch-size", type=int, default=4,
                        help="Batch size for the batched-throughput test (0 = skip).")
    parser.add_argument("--jtop", action="store_true",
                        help="Also log power / GPU load / temperature (needs jetson-stats).")
    parser.add_argument("--output", default="results", help="Parent folder for run folders.")
    parser.add_argument("--run-name", default=None,
                        help="Folder name for this run. Default: <timestamp>_<backbone>_<shots>shot")
    parser.add_argument("--notes", default="", help="Free text stored in the report.")
    cli.add_backbone_args(parser)
    args = parser.parse_args(argv)
    if cli.handle_list_backbones(args):
        return 0

    timestamp = datetime.datetime.now(datetime.timezone.utc)
    safe_backbone = args.backbone.replace(":", "-").replace("/", "-").replace(".", "-")
    run_name = args.run_name or "%s_%s_%dshot" % (
        timestamp.strftime("%Y%m%d-%H%M%S"), safe_backbone, args.shots)

    # 1. start monitors before loading
    mem = profiler.MemorySampler().start()
    jet = profiler.JetsonStatsSampler()
    if args.jtop:
        jet.start()
    t_run_start = profiler.now()

    # 2. load model
    print("[1/5] Loading backbone '%s' ..." % args.backbone)
    backbone = cli.build_backbone(args)
    mem.mark("after_model_load")
    print("      %s | %s | %s | embed dim %d | loaded in %.2f s" % (
        backbone.spec, backbone.device, backbone.precision,
        backbone.embed_dim, backbone.load_time_s))
    backbone.reset_gpu_peak()

    # 3. enrol labelled examples
    print("[2/5] Enrolling support images (%s shots per class) ..." % (args.shots or "all"))
    t0 = profiler.now()
    clf, chosen = enroll(backbone, args.support, args.shots, args.seed,
                         method=args.method, k=args.k,
                         unknown_threshold=args.unknown_threshold)
    enroll_time_s = profiler.now() - t0
    num_support = sum(len(v) for v in chosen.values())
    class_names = clf.class_names

    # gather query images + their true lables
    query_by_class = data.list_images_by_class(args.query)
    query_items = [] # path, true class name
    for name in sorted(query_by_class):
        if name not in class_names:
            print("      WARNING: query class '%s' has no support examples; skipped." % name)
            continue
        query_items.extend((p, name) for p in query_by_class[name])
    if not query_items:
        sys.exit("No query images whose class also exists in the support folder.")
    # prevent same file from being both exmple and test image
    support_names = {os.path.basename(p) for v in chosen.values() for p in v}
    overlap = [p for p, _ in query_items if os.path.basename(p) in support_names
               and os.path.abspath(args.query) == os.path.abspath(args.support)]
    if overlap:
        print("      WARNING: support and query folders are the same, accuracy will be inflated.")

    print("[3/5] Warm-up (%d untimed inferences) ..." % args.warmup)
    warm = data.load_and_preprocess(query_items[0][0], backbone)[np.newaxis]
    for _ in range(args.warmup):
        backbone.embed(warm)

    # 4. timed single-image loop
    print("[4/5] Timing %d query images x %d repeats ..." % (len(query_items), args.repeats))
    t_pre, t_inf, t_cls, t_all = [], [], [], []
    per_image_rows = []
    tensors_cache = [] # preprocessed images reused by the batch test
    t_loop_start = profiler.now()
    for rep in range(args.repeats):
        for path, true_name in query_items:
            start = profiler.now()
            x = data.load_and_preprocess(path, backbone)
            a = profiler.now()
            embedding = backbone.embed(x[np.newaxis])
            b = profiler.now()
            pred = clf.predict(embedding)[0]
            end = profiler.now()

            pre_ms, inf_ms = (a - start) * 1000.0, (b - a) * 1000.0
            cls_ms, all_ms = (end - b) * 1000.0, (end - start) * 1000.0
            t_pre.append(pre_ms); t_inf.append(inf_ms)
            t_cls.append(cls_ms); t_all.append(all_ms)
            if rep == 0:
                tensors_cache.append(x)
            per_image_rows.append({
                "repeat": rep,
                "image": path,
                "true_label": true_name,
                "predicted_label": pred["label"],
                "correct": int(pred["label"] == true_name),
                "score": pred["score"],
                "margin": pred["margin"],
                "load_and_preprocess_ms": round(pre_ms, 3),
                "inference_ms": round(inf_ms, 3),
                "classify_ms": round(cls_ms, 3),
                "end_to_end_ms": round(all_ms, 3),
            })
    loop_elapsed_s = profiler.now() - t_loop_start
    num_timed = len(t_all)

    # 5. batched throughput
    batched = {}
    if args.throughput_batch_size and args.throughput_batch_size > 0:
        print("[5/5] Batched throughput (batch size %d) ..." % args.throughput_batch_size)
        batched = measure_batched_throughput(
            backbone, tensors_cache, args.throughput_batch_size, args.repeats)

    # stop monitors
    gpu_mem = backbone.gpu_memory_mb()
    memory_report = mem.stop()
    memory_report.update(gpu_mem)
    jetson_report = jet.stop() if args.jtop else {"available": False, "note": "run without --jtop"}
    total_run_s = profiler.now() - t_run_start

    # accuracy
    first_pass = [r for r in per_image_rows if r["repeat"] == 0]
    pred_labels = class_names + [UNKNOWN_LABEL]
    confusion = [[0] * len(pred_labels) for _ in class_names]
    for r in first_pass:
        confusion[class_names.index(r["true_label"])][pred_labels.index(r["predicted_label"])] += 1
    if all(row[-1] == 0 for row in confusion):
        pred_labels = pred_labels[:-1]
        confusion = [row[:-1] for row in confusion]
    per_class = {}
    for i, name in enumerate(class_names):
        total = sum(confusion[i])
        correct = confusion[i][i]
        per_class[name] = {"correct": correct, "total": total,
                           "accuracy_percent": round(100.0 * correct / total, 2) if total else None}
    num_correct = sum(r["correct"] for r in first_pass)

    # metrics dictionary
    metrics = {
        "run": {
            "name": run_name,
            "timestamp_utc": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "command": " ".join(sys.argv),
            "notes": args.notes,
            "total_run_time_s": round(total_run_s, 2),
        },
        "system": profiler.system_info(),
        "config": {
            "backbone_argument": args.backbone,
            "pretrained": not args.no_pretrained,
            "shots_per_class_requested": args.shots,
            "shots_per_class_used": "/".join(str(len(chosen[c])) for c in class_names)
                                    if len({len(v) for v in chosen.values()}) > 1
                                    else str(len(chosen[class_names[0]])),
            "seed": args.seed,
            "method": args.method,
            "knn_k": args.k if args.method == "knn" else None,
            "unknown_threshold": args.unknown_threshold,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "model": backbone.describe(),
        "dataset": {
            "classes": class_names,
            "support_dir": args.support,
            "query_dir": args.query,
            "num_support_images": num_support,
            "num_query_images": len(query_items),
            "support_images_used": {c: [os.path.basename(p) for p in chosen[c]] for c in class_names},
        },
        "accuracy": {
            "top1_accuracy_percent": round(100.0 * num_correct / len(first_pass), 2),
            "num_correct": num_correct,
            "num_evaluated": len(first_pass),
            "per_class": per_class,
            "confusion_matrix": {"true_labels": class_names,
                                 "predicted_labels": pred_labels,
                                 "counts": confusion},
        },
        "enrollment": {
            "total_time_s": round(enroll_time_s, 3),
            "time_per_support_image_ms": round(1000.0 * enroll_time_s / num_support, 2),
        },
        "latency_ms": {
            "description": "Per-image latency at batch size 1, in milliseconds.",
            "load_and_preprocess": profiler.summarize_ms(t_pre),
            "inference": profiler.summarize_ms(t_inf),
            "classify": profiler.summarize_ms(t_cls),
            "end_to_end": profiler.summarize_ms(t_all),
        },
        "throughput": {
            "description": "Images processed per second.",
            # Wall-clock over the whole timed loop so loop overhead is included
            "end_to_end_images_per_s": round(num_timed / loop_elapsed_s, 3),
            "inference_only_images_per_s": round(1000.0 * num_timed / sum(t_inf), 3),
            "batched": batched,
        },
        "memory_mb": memory_report,
        "jetson_stats": jetson_report,
    }

    out_dir = os.path.join(args.output, run_name)
    paths = report.write_run_outputs(out_dir, metrics, per_image_rows)

    # console summary
    lat = metrics["latency_ms"]
    print("")
    print("================= RESULT: %s =================" % run_name)
    print("Accuracy (top-1)            : %.1f %%  (%d/%d)" % (
        metrics["accuracy"]["top1_accuracy_percent"], num_correct, len(first_pass)))
    print("Latency end-to-end, median  : %.2f ms   (p95 %.2f ms)" % (
        lat["end_to_end"]["median_ms"], lat["end_to_end"]["p95_ms"]))
    print("Latency inference,  median  : %.2f ms" % lat["inference"]["median_ms"])
    print("Throughput end-to-end       : %.2f images/s" % metrics["throughput"]["end_to_end_images_per_s"])
    if batched.get("images_per_s"):
        print("Throughput batched (b=%d)    : %.2f images/s" % (batched["batch_size"], batched["images_per_s"]))
    print("Peak process memory (RSS)   : %s MB" % memory_report.get("process_rss_peak_mb"))
    print("Peak system memory increase : %s MB" % memory_report.get("system_used_increase_peak_mb"))
    print("Reports written to          : %s" % out_dir)
    for key in ("summary_md", "metrics_json", "per_image_csv"):
        print("    " + paths[key])
    return 0


if __name__ == "__main__":
    sys.exit(main())
