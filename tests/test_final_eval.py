"""Uji evaluasi final: metrik murni, gerbang konfirmasi, dan alur dengan fake."""

from __future__ import annotations

import csv
import importlib
import json

import numpy as np

from src import data_io
from src import final_eval


# ---------------------------------------------------------------------------
# Penjaga test set
# ---------------------------------------------------------------------------


def test_import_does_not_unlock_test_set(monkeypatch):
    monkeypatch.delenv(data_io.ALLOW_TEST_ENV, raising=False)
    importlib.reload(final_eval)
    assert data_io.test_access_allowed() is False


def test_main_requires_confirm_phrase():
    assert final_eval.main([]) == 2
    assert final_eval.main(["--confirm", "salah"]) == 2


# ---------------------------------------------------------------------------
# Metrik murni
# ---------------------------------------------------------------------------


def test_classification_metrics_perfect():
    labels = ["a", "b", "c"]
    m = final_eval.classification_metrics(["a", "b", "c"], ["a", "b", "c"], labels)
    assert m["accuracy"] == 1.0
    assert m["f1_macro"] == 1.0
    assert m["f1_weighted"] == 1.0
    assert m["confusion"]["a"]["a"] == 1


def test_classification_metrics_confusion_and_empty():
    labels = ["a", "b"]
    m = final_eval.classification_metrics(["a", "a", "b"], ["a", "b", "b"], labels)
    assert m["confusion"]["a"]["a"] == 1
    assert m["confusion"]["a"]["b"] == 1
    assert m["confusion"]["b"]["b"] == 1
    # kelas b: 1 benar dari 2 prediksi -> precision 0.5, recall 1.0
    assert m["per_class"]["b"]["precision"] == 0.5
    assert m["per_class"]["b"]["recall"] == 1.0
    assert 0.0 <= m["f1_macro"] <= 1.0

    empty = final_eval.classification_metrics([], [], labels)
    assert empty["accuracy"] == 0.0
    assert empty["per_class"] == {}


# ---------------------------------------------------------------------------
# Laporan
# ---------------------------------------------------------------------------


def _fake_result(**over):
    result = {
        "sandbox": False,
        "model_dir": "models/x",
        "test_path": "test.csv",
        "mixed_path": "mixed_test.csv",
        "temperature": 0.5,
        "tau1": 0.4,
        "intent_metrics": {
            "accuracy": 0.9,
            "f1_macro": 0.88,
            "f1_weighted": 0.89,
            "precision_macro": 0.87,
            "recall_macro": 0.86,
            "per_class": {"a": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 2}},
            "confusion": {"a": {"a": 2}},
        },
        "ece": 0.12,
        "at_tau1": {"coverage": 0.8, "accepted_accuracy": 0.95, "oos_false_accept": 0.0},
        "route_metrics": {
            "accuracy": 0.75,
            "n": 4,
            "per_route": {"faq": {"precision": 1.0, "recall": 0.75, "f1": 0.85, "support": 4}},
            "confusion": {"faq": {"faq": 3, "fallback": 1}},
        },
    }
    result.update(over)
    return result


def test_render_report_sections():
    text = final_eval.render_report(_fake_result())
    assert "# Laporan Evaluasi FINAL" in text
    assert "Klasifikasi intent (test)" in text
    assert "Routing end-to-end (test)" in text
    assert "f1_macro" in text


def test_render_report_without_e2e():
    text = final_eval.render_report(_fake_result(route_metrics=None, mixed_path=None))
    assert "mixed_test.csv` tidak ditemukan" in text


def test_render_report_sandbox_tag():
    text = final_eval.render_report(_fake_result(sandbox=True))
    assert "[SANDBOX]" in text


# ---------------------------------------------------------------------------
# Alur end-to-end dengan fake (tanpa model, tanpa test set proyek)
# ---------------------------------------------------------------------------


def _write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def _prepare(tmp_path, cfg):
    intents = data_io.get_intents(cfg)
    labels = intents[:3]

    test_csv = tmp_path / "test.csv"
    _write_csv(
        test_csv,
        ["text", "intent"],
        [[f"contoh {lab}", lab] for lab in labels],
    )

    mixed_csv = tmp_path / "mixed_test.csv"
    _write_csv(
        mixed_csv,
        ["question", "expected_route"],
        [[f"tanya {i}", "faq"] for i in range(3)],
    )

    model = tmp_path / "model"
    model.mkdir()
    (model / "calibration.json").write_text(
        json.dumps({"temperature": 1.0, "tau1": 0.5}), encoding="utf-8"
    )

    def fake_logits(model_dir, df, max_length, batch_size):
        label2id = {name: i for i, name in enumerate(intents)}
        id2label = {i: name for name, i in label2id.items()}
        n = len(intents)
        logits, truths = [], []
        for intent in df["intent"]:
            idx = label2id[intent]
            row = np.zeros(n)
            row[idx] = 5.0
            logits.append(row)
            truths.append(idx)
        return np.array(logits), np.array(truths, dtype=np.int64), id2label

    class FakeRouter:
        def __init__(self, *_args, **_kwargs):
            pass

        def route(self, text):
            from src.router import RouteDecision

            return RouteDecision(
                route="faq", intent=labels[0], confidence=0.9, answer="x", reason="faq_answer"
            )

    def fake_router_factory(*_args, **_kwargs):
        return FakeRouter()

    return test_csv, mixed_csv, model, fake_logits, fake_router_factory


def test_run_final_eval_end_to_end_with_fakes(tmp_path, monkeypatch):
    monkeypatch.delenv(data_io.ALLOW_TEST_ENV, raising=False)
    cfg = data_io.load_config()
    test_csv, mixed_csv, model, fake_logits, fake_router = _prepare(tmp_path, cfg)

    result = final_eval.run_final_eval(
        cfg,
        model_dir=model,
        vectorstore_dir=None,
        answers_path=tmp_path / "intents.yaml",
        sandbox=False,
        test_path=test_csv,
        mixed_path=mixed_csv,
        router_factory=fake_router,
        logits_fn=fake_logits,
    )

    assert result["temperature"] == 1.0
    assert result["tau1"] == 0.5
    assert result["intent_metrics"]["accuracy"] == 1.0
    assert result["route_metrics"]["accuracy"] == 1.0
    # kunci test set dikembalikan setelah selesai
    assert data_io.test_access_allowed() is False


def test_run_final_eval_skips_missing_mixed(tmp_path, monkeypatch):
    monkeypatch.delenv(data_io.ALLOW_TEST_ENV, raising=False)
    cfg = data_io.load_config()
    test_csv, _mixed, model, fake_logits, fake_router = _prepare(tmp_path, cfg)
    missing = tmp_path / "tidak_ada" / "mixed_test.csv"

    result = final_eval.run_final_eval(
        cfg,
        model_dir=model,
        vectorstore_dir=None,
        answers_path=tmp_path / "intents.yaml",
        sandbox=True,
        test_path=test_csv,
        mixed_path=missing,
        router_factory=fake_router,
        logits_fn=fake_logits,
    )
    assert result["route_metrics"] is None
    assert result["mixed_path"] is None
