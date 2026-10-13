"""Uji bagian murni dari ``src/train.py`` (tanpa mengunduh model / jaringan)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src import train


class _EP:
    """Tiruan ``EvalPrediction`` (punya atribut predictions & label_ids)."""

    def __init__(self, predictions, label_ids):
        self.predictions = predictions
        self.label_ids = label_ids


# ---------------------------------------------------------------------------
# Peta label
# ---------------------------------------------------------------------------


def test_make_label_maps_deterministic_order():
    intents = ["tanya_daftar", "tanya_biaya", "lainnya"]
    label2id, id2label = train.make_label_maps(intents)
    assert label2id == {"tanya_daftar": 0, "tanya_biaya": 1, "lainnya": 2}
    assert id2label == {0: "tanya_daftar", 1: "tanya_biaya", 2: "lainnya"}


def test_model_slug_strips_namespace():
    assert train.model_slug("cahya/distilbert-base-indonesian") == "distilbert-base-indonesian"
    assert train.model_slug("distilbert/distilbert-base-multilingual-cased") == (
        "distilbert-base-multilingual-cased"
    )


# ---------------------------------------------------------------------------
# Ekstraksi prediksi (kompatibilitas transformers 5.x)
# ---------------------------------------------------------------------------


def test_to_array_plain_2d():
    arr = np.zeros((4, 3))
    assert train._to_array(arr).shape == (4, 3)


def test_to_array_ignores_nested_hidden_states():
    logits = np.zeros((6, 3))
    hidden = tuple(np.zeros((6, 8, 16)) for _ in range(7))
    out = train._to_array((logits, hidden))
    assert out.shape == (6, 3)


def test_to_array_concatenates_per_batch_arrays():
    a = np.zeros((4, 3))
    b = np.ones((2, 3))
    out = train._to_array((a, b))
    assert out.shape == (6, 3)
    assert out[4].tolist() == [1.0, 1.0, 1.0]


def test_to_array_none_is_empty():
    assert train._to_array(None).size == 0


# ---------------------------------------------------------------------------
# Metrik
# ---------------------------------------------------------------------------


def test_compute_metrics_perfect_predictions():
    logits = np.array([[2.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
    labels = np.array([0, 1, 0])
    res = train.compute_metrics(_EP(logits, labels))
    assert res["accuracy"] == pytest.approx(1.0)
    assert res["f1_macro"] == pytest.approx(1.0)


def test_compute_metrics_with_tuple_nested_predictions():
    logits = np.array([[2.0, 0.0], [0.0, 2.0]])
    hidden = tuple(np.zeros((2, 8, 16)) for _ in range(7))
    res = train.compute_metrics(_EP((logits, hidden), np.array([0, 1])))
    assert res["accuracy"] == pytest.approx(1.0)


def test_compute_metrics_plain_tuple_fallback():
    logits = np.array([[0.0, 1.0], [5.0, 0.0]])
    labels = np.array([1, 0])
    res = train.compute_metrics((logits, labels))
    assert res["accuracy"] == pytest.approx(1.0)


def test_compute_metrics_accepts_extra_kwargs():
    logits = np.array([[2.0, 0.0]])
    labels = np.array([0])
    res = train.compute_metrics(_EP(logits, labels), compute_result=True)
    assert res["accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Bobot kelas & warmup
# ---------------------------------------------------------------------------


def test_class_weights_balanced_inverse_frequency():
    w = train.class_weights_from_labels([0, 0, 0, 1], num_labels=2)
    assert w.shape == (2,)
    # Kelas 1 lebih jarang -> bobot lebih besar.
    assert w[1] > w[0]


def test_class_weights_handles_missing_class():
    w = train.class_weights_from_labels([0, 0], num_labels=3)
    assert w.shape == (3,)
    assert np.isfinite(w.numpy()).all()


def test_warmup_steps_ratio_conversion():
    # 100 sampel, batch 10 -> 10 langkah/epoch, 2 epoch = 20 langkah; 10% -> 2.
    assert train.warmup_steps_for(0.1, 100, 10, 2) == 2
    assert train.warmup_steps_for(0.0, 100, 10, 2) == 0


# ---------------------------------------------------------------------------
# Agregasi multi-seed
# ---------------------------------------------------------------------------


def _result(acc, f1):
    return train.TrainResult(
        model_name="m",
        seed=42,
        lr=3e-5,
        batch=16,
        epochs=1,
        metrics={"accuracy": acc, "f1_macro": f1},
        duration_s=1.0,
        output_dir="x",
    )


def test_aggregate_mean_std():
    agg = train.aggregate([_result(0.8, 0.7), _result(0.6, 0.5)])
    assert agg["f1_macro_mean"] == pytest.approx(0.6)
    assert agg["accuracy_mean"] == pytest.approx(0.7)
    assert agg["n_runs"] == 2


def test_aggregate_empty():
    assert train.aggregate([]) == {}


def test_publish_best_copies_files(tmp_path):
    src = tmp_path / "run"
    src.mkdir()
    (src / "config.json").write_text("{}", encoding="utf-8")
    (src / "model.safetensors").write_bytes(b"weights")
    (src / "label_map.json").write_text("{}", encoding="utf-8")
    res = train.TrainResult("m", 42, 3e-5, 16, 1, {"f1_macro": 0.9}, 1.0, str(src))

    publish_dir = tmp_path / "published"
    train.publish_best(res, publish_dir)

    assert (publish_dir / "model.safetensors").read_bytes() == b"weights"
    assert (publish_dir / "selected.json").exists()


# ---------------------------------------------------------------------------
# Orkestrasi search / multiseed (train_single di-mock)
# ---------------------------------------------------------------------------


def _tcfg():
    return {
        "learning_rates": [2e-5, 3e-5],
        "batch_sizes": [16, 32],
        "epochs_max": 10,
        "max_length": 64,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "early_stopping_patience": 3,
        "metric_for_best_model": "f1_macro",
        "torch_num_threads": 1,
    }


def _fake_single(f1_by_batch):
    def fake(**kwargs):
        f1 = f1_by_batch[str(kwargs["batch"])]
        return train.TrainResult(
            kwargs["model_name"], kwargs["seed"], kwargs["lr"], kwargs["batch"],
            kwargs["epochs"], {"accuracy": f1, "f1_macro": f1}, 1.0,
            str(kwargs["output_dir"]),
        )
    return fake


def test_run_search_explores_grid_and_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "train_single", _fake_single({"16": 0.5, "32": 0.7}))
    logs = []
    monkeypatch.setattr(train.experiment_log, "log_experiment", lambda **kw: logs.append(kw))

    results = train.run_search(
        model_name="m", train_df=None, val_df=None, intents=["a", "b"],
        tcfg=_tcfg(), seed=42, epochs=1, runs_root=tmp_path / "runs",
        sandbox=True, use_class_weight=False,
    )
    assert len(results) == 4  # 2 lr x 2 batch
    assert len(logs) == 4
    best = max(results, key=lambda r: r.metrics["f1_macro"])
    assert best.batch == 32


def test_run_multiseed_uses_all_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "train_single", _fake_single({"16": 0.6, "32": 0.6}))
    logs = []
    monkeypatch.setattr(train.experiment_log, "log_experiment", lambda **kw: logs.append(kw))

    results = train.run_multiseed(
        model_name="m", train_df=None, val_df=None, intents=["a", "b"],
        tcfg=_tcfg(), lr=3e-5, batch=16, seeds=[42, 43, 44], epochs=1,
        runs_root=tmp_path / "runs", sandbox=True, use_class_weight=False,
    )
    assert [r.seed for r in results] == [42, 43, 44]
    assert len(logs) == 3
    assert train.aggregate(results)["n_runs"] == 3


def test_main_single_mode_wires(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "train_single", _fake_single({"16": 0.5, "32": 0.5}))
    monkeypatch.setattr(train.experiment_log, "log_experiment", lambda **kw: None)
    rc = train.main(["--mode", "single", "--sandbox", "--batch", "16",
                     "--output", str(tmp_path / "out")])
    assert rc == 0


def test_main_search_mode_selects_best(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "train_single", _fake_single({"16": 0.5, "32": 0.9}))
    monkeypatch.setattr(train.experiment_log, "log_experiment", lambda **kw: None)
    out = tmp_path / "out"
    rc = train.main(["--mode", "search", "--sandbox", "--epochs", "1", "--output", str(out)])
    assert rc == 0
    selected = json.loads((out / "selected.json").read_text(encoding="utf-8"))
    assert selected["batch"] == 32


def test_main_multiseed_mode_wires(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "train_single", _fake_single({"16": 0.6, "32": 0.6}))
    monkeypatch.setattr(train.experiment_log, "log_experiment", lambda **kw: None)
    out = tmp_path / "out"
    rc = train.main(["--mode", "multiseed", "--sandbox", "--epochs", "1",
                     "--lr", "3e-5", "--batch", "16", "--output", str(out)])
    assert rc == 0
    assert (out / "selected.json").exists()
