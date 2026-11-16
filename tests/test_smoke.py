"""
test_smoke.py - automated checks
"""

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fewshot import profiler # noqa: E402
from fewshot.classifier import FewShotClassifier # noqa: E402
from fewshot.backbones import l2_normalize, resolve_spec # noqa: E402


def _toy_embeddings(seed=0, per_class=5, dim=16, noise=0.05):
    """Three well-separated clusters standing in for three object classes."""
    rng = np.random.RandomState(seed)
    centres = np.eye(3, dim)
    x = np.concatenate([c + noise * rng.randn(per_class, dim) for c in centres])
    y = np.repeat(np.arange(3), per_class)
    return l2_normalize(x).astype(np.float32), y


@pytest.mark.parametrize("method", ["prototype", "knn"])
def test_classifier_separates_clusters(method):
    support, labels = _toy_embeddings(seed=0)
    query, truth = _toy_embeddings(seed=1)
    clf = FewShotClassifier(method=method).fit(support, labels, ["mug", "water_bottle", "pen"])
    predicted = [p["class_index"] for p in clf.predict(query)]
    assert predicted == list(truth)


def test_unknown_threshold_rejects_unrelated_vector():
    support, labels = _toy_embeddings()
    clf = FewShotClassifier(unknown_threshold=0.5).fit(support, labels, ["a", "b", "c"])
    stranger = np.zeros((1, 16), dtype=np.float32)
    stranger[0, 10] = 1.0 # orthogonal to every class
    assert clf.predict(stranger)[0]["label"] == "unknown"


def test_classifier_save_and_load_roundtrip(tmp_path):
    support, labels = _toy_embeddings()
    clf = FewShotClassifier(method="knn", k=3).fit(support, labels, ["a", "b", "c"], "timm:x")
    path = str(tmp_path / "enrol.npz")
    clf.save(path)
    loaded = FewShotClassifier.load(path)
    assert loaded.class_names == ["a", "b", "c"]
    assert loaded.backbone_spec == "timm:x"
    assert np.allclose(loaded.prototypes, clf.prototypes)


def test_latency_summary_fields_and_values():
    stats = profiler.summarize_ms([10.0, 20.0, 30.0, 40.0])
    assert stats["count"] == 4
    assert stats["mean_ms"] == 25.0 and stats["median_ms"] == 25.0
    assert stats["min_ms"] == 10.0 and stats["max_ms"] == 40.0
    assert stats["p95_ms"] <= stats["p99_ms"] <= stats["max_ms"]


def test_preset_resolution():
    assert resolve_spec("dinov2_vits14").startswith("timm:vit_small_patch14_dinov2")
    assert resolve_spec("convnext_tiny") == "timm:convnext_tiny" # bare timm name
    assert resolve_spec("onnx:m.onnx") == "onnx:m.onnx" # passes thru


def test_end_to_end_benchmark_writes_labelled_reports(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    from scripts import make_demo_data, benchmark, compare_runs

    data_dir = str(tmp_path / "demo")
    out_dir = str(tmp_path / "results")
    make_demo_data.main(["--out", data_dir, "--support-per-class", "3", "--query-per-class", "2"])
    rc = benchmark.main([
        "--support", os.path.join(data_dir, "support"),
        "--query", os.path.join(data_dir, "query"),
        "--backbone", "timm:resnet18", "--no-pretrained", "--device", "cpu",
        "--img-size", "96", "--shots", "3", "--warmup", "1", "--repeats", "2",
        "--throughput-batch-size", "2", "--output", out_dir, "--run-name", "smoke",
    ])
    assert rc == 0

    run_dir = os.path.join(out_dir, "smoke")
    for name in ("metrics.json", "summary.md", "per_image.csv"):
        assert os.path.isfile(os.path.join(run_dir, name))

    with open(os.path.join(run_dir, "metrics.json")) as f:
        m = json.load(f)
    # 3 classes x 2 query images x 2 repeats = 12 timed images
    assert m["latency_ms"]["end_to_end"]["count"] == 12
    assert m["latency_ms"]["inference"]["median_ms"] > 0
    assert m["throughput"]["end_to_end_images_per_s"] > 0
    assert m["throughput"]["batched"]["images_per_s"] > 0
    assert m["memory_mb"]["process_rss_peak_mb"] >= m["memory_mb"]["process_rss_baseline_mb"]
    assert m["accuracy"]["num_evaluated"] == 6
    assert m["dataset"]["classes"] == ["mug", "pen", "water_bottle"]

    assert compare_runs.main(["--results", out_dir]) == 0
    assert os.path.isfile(os.path.join(out_dir, "comparison.csv"))
