"""Uji bagian murni dari ``src/calibrate.py`` (tanpa model / jaringan)."""

from __future__ import annotations

import numpy as np
import pytest

from src import calibrate


def test_softmax_rows_sum_to_one():
    logits = np.array([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    probs = calibrate.softmax_with_temperature(logits, 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_higher_temperature_is_more_uniform():
    logits = np.array([[5.0, 0.0, 0.0]])
    sharp = calibrate.softmax_with_temperature(logits, 0.5)
    soft = calibrate.softmax_with_temperature(logits, 5.0)
    assert soft.max() < sharp.max()


def test_fit_temperature_smooths_overconfident_logits():
    # Model selalu yakin kelas 0 tetapi sering salah -> perlu diperlembut (T>1).
    labels = np.array([0, 1, 2, 0, 1, 2])
    logits = np.zeros((6, 3))
    logits[:, 0] = 6.0

    temp = calibrate.fit_temperature(logits, labels, t_min=0.05, t_max=20.0, points=500)
    assert temp > 1.0  # perlu melembutkan

    def nll(t):
        p = calibrate.softmax_with_temperature(logits, t)
        return -np.log(p[np.arange(len(labels)), labels]).mean()

    assert nll(temp) <= nll(1.0) + 1e-6


def test_expected_calibration_error_zero_for_perfect():
    # Probabilitas satu-hot tepat -> akurasi 1 dan keyakinan 1 -> ECE 0.
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    assert calibrate.expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(0.0)


def test_expected_calibration_error_positive_when_miscalibrated():
    # Selalu yakin 0.99 tetapi hanya 50% benar.
    probs = np.array([[0.99, 0.01], [0.99, 0.01], [0.01, 0.99], [0.01, 0.99]])
    labels = np.array([0, 1, 0, 1])
    assert calibrate.expected_calibration_error(probs, labels, n_bins=10) > 0.3


def test_tau_values_grid():
    assert calibrate.tau_values(0.4, 0.6, 0.05) == [0.4, 0.45, 0.5, 0.55, 0.6]


def _probs_for_tau():
    # 4 sampel, intent OOS = indeks 2.
    probs = np.array([
        [0.90, 0.05, 0.05],  # pred 0, benar
        [0.60, 0.30, 0.10],  # pred 0, benar
        [0.20, 0.70, 0.10],  # pred 1, benar
        [0.50, 0.30, 0.20],  # OOS asli, pred salah
    ])
    labels = np.array([0, 0, 1, 2])
    return probs, labels


def test_metrics_at_tau_excludes_low_confidence():
    probs, labels = _probs_for_tau()
    m = calibrate.metrics_at_tau(probs, labels, 0.6, oos_idx=2)
    assert m["n_accepted"] == 3
    assert m["coverage"] == pytest.approx(0.75)
    assert m["accepted_accuracy"] == pytest.approx(1.0)
    assert m["oos_false_accept"] == pytest.approx(0.0)


def test_metrics_at_tau_includes_oos_false_accept():
    probs, labels = _probs_for_tau()
    m = calibrate.metrics_at_tau(probs, labels, 0.5, oos_idx=2)
    assert m["n_accepted"] == 4
    assert m["accepted_accuracy"] == pytest.approx(0.75)
    assert m["oos_false_accept"] == pytest.approx(1.0)


def test_choose_tau1_picks_eligible_max_coverage():
    probs, labels = _probs_for_tau()
    table = [calibrate.metrics_at_tau(probs, labels, t, 2) for t in (0.5, 0.6)]
    chosen, status = calibrate.choose_tau1(table, target_acc=0.95, target_oos=0.05)
    assert status == "ok"
    assert chosen["tau"] == pytest.approx(0.6)


def test_choose_tau1_relaxed_when_acc_unreachable():
    table = [
        {"tau": 0.5, "coverage": 1.0, "accepted_accuracy": 0.5, "n_accepted": 4,
         "oos_false_accept": 0.0, "n_oos": 1},
        {"tau": 0.9, "coverage": 0.1, "accepted_accuracy": 0.7, "n_accepted": 1,
         "oos_false_accept": 0.0, "n_oos": 1},
    ]
    chosen, status = calibrate.choose_tau1(table, target_acc=0.99, target_oos=0.05)
    assert status == "relaxed"
    # Semua memenuhi batas OOS -> pilih akurasi tertinggi.
    assert chosen["tau"] == pytest.approx(0.9)


def test_choose_tau1_fallback_min_oos():
    table = [
        {"tau": 0.5, "coverage": 1.0, "accepted_accuracy": 0.9, "n_accepted": 4,
         "oos_false_accept": 0.5, "n_oos": 2},
        {"tau": 0.9, "coverage": 0.5, "accepted_accuracy": 0.9, "n_accepted": 2,
         "oos_false_accept": 0.3, "n_oos": 2},
    ]
    chosen, status = calibrate.choose_tau1(table, target_acc=0.99, target_oos=0.05)
    assert status == "relaxed"
    assert chosen["tau"] == pytest.approx(0.9)  # OOS terkecil
