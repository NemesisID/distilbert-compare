"""Evaluasi FINAL pada test set terkunci (HANYA modul ini yang boleh).

Modul ini adalah **satu-satunya** tempat yang menyetel ``ALLOW_TEST=1`` untuk
membuka test set. Karena itu, modul ini hanya dijalankan saat pemilik proyek
memberi perintah eksplisit:

    python -m src.final_eval --confirm "JALANKAN EVALUASI FINAL"
    python -m src.final_eval --sandbox --confirm "JALANKAN EVALUASI FINAL"

Yang dihitung di test set (sekali, tanpa menyetel ulang apa pun):
- klasifikasi intent: accuracy, precision/recall/F1 (macro & weighted), F1 per kelas;
- kalibrasi: ECE pada suhu terkalibrasi + coverage/accepted-accuracy pada tau1;
- routing end-to-end (bila ``mixed_test.csv`` ada): akurasi route + metrik per-route.

Menghasilkan ``reports/final_eval.md`` (atau ``reports/sandbox_final_eval.md``)
dan mencatatnya ke ``EXPERIMENTS.md``. Setelan T/tau1 dibaca dari
``models/<slug>/calibration.json`` (diperoleh dari val) — TIDAK dipilih ulang.

Catatan: mengimpor modul ini TIDAK membuka test set; kunci hanya dibuka di
dalam :func:`run_final_eval` lalu dikembalikan seperti semula.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

from src import data_io

#: Frasa konfirmasi yang wajib diketik persis untuk menjalankan evaluasi final.
CONFIRM_PHRASE = "JALANKAN EVALUASI FINAL"


# ---------------------------------------------------------------------------
# Metrik murni (mudah diuji)
# ---------------------------------------------------------------------------


def classification_metrics(y_true: list, y_pred: list, labels: list) -> dict:
    """Hitung accuracy, P/R/F1 makro & terbobot, F1 per kelas, dan confusion.

    ``y_true``/``y_pred``/``labels`` berupa nama label (string), agar laporan
    tidak perlu lagi memetakan indeks.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
    )

    if not y_true:
        return {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "per_class": {},
            "confusion": {},
        }

    accuracy = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    _, _, f_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }
    confusion = {a: {b: 0 for b in labels} for a in labels}
    for actual, predicted in zip(y_true, y_pred):
        confusion[actual][predicted] += 1

    return {
        "accuracy": accuracy,
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f_macro),
        "f1_weighted": float(f_weighted),
        "per_class": per_class,
        "confusion": confusion,
    }


def _load_calibration(model_dir: Path, cfg: dict) -> tuple[float, float]:
    """Baca T & tau1 dari ``calibration.json`` (dari val); fallback konservatif."""
    cal_path = Path(model_dir) / "calibration.json"
    if cal_path.exists():
        data = json.loads(cal_path.read_text(encoding="utf-8"))
        return float(data["temperature"]), float(data["tau1"])
    warnings.warn(
        f"{cal_path} tidak ada; pakai T=1.0 dan tau1={cfg['calibration']['tau1_max']}.",
        stacklevel=2,
    )
    return 1.0, float(cfg["calibration"]["tau1_max"])


# ---------------------------------------------------------------------------
# Evaluasi
# ---------------------------------------------------------------------------


def run_final_eval(
    cfg: dict,
    *,
    model_dir: str | Path,
    vectorstore_dir: str | Path | None,
    answers_path: str | Path,
    method: str = "rrf",
    sandbox: bool = False,
    limit: int | None = None,
    test_path: str | Path | None = None,
    mixed_path: str | Path | None = None,
    router_factory=None,
    logits_fn=None,
) -> dict:
    """Jalankan evaluasi final pada test set (membuka kunci ``ALLOW_TEST`` sementara).

    Parameter ``router_factory`` dan ``logits_fn`` dapat diganti (dependency
    injection) untuk pengujian tanpa memuat model.
    """
    from src import calibrate
    from src.router import Router
    from src.router import evaluate as evaluate_routes

    router_factory = router_factory or Router
    logits_fn = logits_fn or calibrate.compute_logits

    previous = os.environ.get(data_io.ALLOW_TEST_ENV)
    os.environ[data_io.ALLOW_TEST_ENV] = "1"
    try:
        intents = data_io.get_intents(cfg)
        cal_cfg = cfg["calibration"]
        oos_idx = intents.index(str(cfg["routing"]["oos_intent"]))

        if test_path is None:
            if sandbox:
                base = data_io.resolve_path(cfg["paths"]["sandbox"]).parent
                test_path = base / "processed" / "test.csv"
            else:
                test_path = data_io.resolve_path(cfg["paths"]["processed_dir"]) / "test.csv"
        if mixed_path is None:
            if sandbox:
                base = data_io.resolve_path(cfg["paths"]["sandbox"]).parent
                mixed_path = base / "e2e" / "mixed_test.csv"
            else:
                mixed_path = data_io.resolve_path(cfg["paths"]["e2e_dir"]) / "mixed_test.csv"

        intent_df = data_io.read_csv(test_path, data_io.INTENT_TABLE, allow_test=True)
        if limit:
            intent_df = intent_df.head(limit)

        logits, labels_true, id2label = logits_fn(
            Path(model_dir),
            intent_df,
            int(cfg["training"]["max_length"]),
            int(cal_cfg["batch_size"]),
        )
        temperature, tau1 = _load_calibration(Path(model_dir), cfg)

        probs = calibrate.softmax_with_temperature(logits, temperature)
        y_pred_idx = probs.argmax(axis=1)
        label_names = [id2label[i] for i in range(len(id2label))]
        intent_metrics = classification_metrics(
            [id2label[int(i)] for i in labels_true],
            [id2label[int(i)] for i in y_pred_idx],
            label_names,
        )
        ece = calibrate.expected_calibration_error(probs, labels_true, int(cal_cfg["n_bins"]))
        at_tau1 = calibrate.metrics_at_tau(probs, labels_true, tau1, oos_idx)

        route_metrics = None
        used_mixed: str | None = None
        mixed = Path(mixed_path)
        if mixed.exists():
            mixed_df = data_io.read_csv(mixed, data_io.MIXED_TABLE, allow_test=True)
            if limit:
                mixed_df = mixed_df.head(limit)
            router = router_factory(
                cfg,
                model_dir=model_dir,
                vectorstore_dir=vectorstore_dir,
                answers_path=answers_path,
                method=method,
            )
            route_metrics = evaluate_routes(
                router, mixed_df.to_dict("records"), [str(r) for r in cfg["routing"]["routes"]]
            )
            used_mixed = str(mixed)

        return {
            "sandbox": sandbox,
            "model_dir": str(model_dir),
            "test_path": str(test_path),
            "mixed_path": used_mixed,
            "temperature": temperature,
            "tau1": tau1,
            "intent_metrics": intent_metrics,
            "ece": ece,
            "at_tau1": at_tau1,
            "route_metrics": route_metrics,
        }
    finally:
        if previous is None:
            os.environ.pop(data_io.ALLOW_TEST_ENV, None)
        else:
            os.environ[data_io.ALLOW_TEST_ENV] = previous


# ---------------------------------------------------------------------------
# Laporan
# ---------------------------------------------------------------------------


def render_report(result: dict) -> str:
    """Susun laporan Markdown hasil evaluasi final."""
    m = result["intent_metrics"]
    tag = "[SANDBOX] " if result["sandbox"] else ""
    lines = [
        "# Laporan Evaluasi FINAL (test set)",
        "",
        f"> **{tag}Test set dibaca SEKALI.** T & tau1 dari val (tidak dipilih ulang).",
        "",
        f"- Folder model: `{result['model_dir']}`",
        f"- Data test (intent): `{result['test_path']}`",
        f"- Data test (e2e): `{result['mixed_path'] or 'tidak ada'}`",
        f"- Suhu terkalibrasi **T = {result['temperature']:.4f}**",
        f"- Ambang tau1 = {result['tau1']:.4f}",
        "",
        "## Klasifikasi intent (test)",
        "",
        f"- accuracy = **{m['accuracy']:.4f}**",
        f"- f1_macro = **{m['f1_macro']:.4f}**",
        f"- f1_weighted = {m['f1_weighted']:.4f}",
        f"- precision_macro = {m['precision_macro']:.4f}",
        f"- recall_macro = {m['recall_macro']:.4f}",
        "",
        "## Kalibrasi (test)",
        "",
        f"- ECE (T terkalibrasi) = {result['ece']:.4f}",
        f"- coverage@tau1 = {result['at_tau1']['coverage']:.4f}",
        f"- accepted_accuracy@tau1 = {result['at_tau1']['accepted_accuracy']:.4f}",
        f"- oos_false_accept@tau1 = {result['at_tau1']['oos_false_accept']:.4f}",
        "",
        "## F1 per kelas (test)",
        "",
        "| intent | precision | recall | f1 | support |",
        "|--------|-----------|--------|----|---------|",
    ]
    for label, pm in m["per_class"].items():
        lines.append(
            f"| {label} | {pm['precision']:.4f} | {pm['recall']:.4f} "
            f"| {pm['f1']:.4f} | {pm['support']} |"
        )

    lines += ["", "## Confusion matrix (baris = aktual, kolom = prediksi)", ""]
    if m["confusion"]:
        labels = list(m["confusion"].keys())
        lines.append("| aktual \\ prediksi | " + " | ".join(labels) + " |")
        lines.append("|" + "---|" * (len(labels) + 1))
        for actual in labels:
            cells = " | ".join(str(m["confusion"][actual][p]) for p in labels)
            lines.append(f"| **{actual}** | {cells} |")

    if result["route_metrics"] is not None:
        rm = result["route_metrics"]
        lines += [
            "",
            "## Routing end-to-end (test)",
            "",
            f"- akurasi route = **{rm['accuracy']:.4f}** (n = {rm['n']})",
            "",
            "| route | precision | recall | f1 | support |",
            "|-------|-----------|--------|----|---------|",
        ]
        for label, pm in rm["per_route"].items():
            lines.append(
                f"| {label} | {pm['precision']:.4f} | {pm['recall']:.4f} "
                f"| {pm['f1']:.4f} | {pm['support']} |"
            )
        lines += ["", "### Confusion matrix route", ""]
        labels = list(rm["confusion"].keys())
        lines.append("| aktual \\ prediksi | " + " | ".join(labels) + " |")
        lines.append("|" + "---|" * (len(labels) + 1))
        for actual in labels:
            cells = " | ".join(str(rm["confusion"][actual][p]) for p in labels)
            lines.append(f"| **{actual}** | {cells} |")
    else:
        lines += [
            "",
            "## Routing end-to-end (test)",
            "",
            "> `mixed_test.csv` tidak ditemukan; evaluasi route dilewati.",
        ]

    lines += [
        "",
        "> Catatan: angka ini dihitung SEKALI pada test set. Untuk data sintetis,",
        "> hasil diberi tanda `[SANDBOX]` dan BUKAN hasil penelitian.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_paths(cfg: dict, args: argparse.Namespace) -> dict:
    """Tentukan jalur model/vectorstore/jawaban (asli atau sandbox)."""
    from src.train import model_slug

    if args.sandbox:
        base = data_io.resolve_path(cfg["paths"]["sandbox"]).parent
        dense = base / "vectorstore_dense"
        return {
            "model_dir": data_io.resolve_path(args.model_dir)
            if args.model_dir
            else data_io.resolve_path(cfg["paths"]["models_dir"]) / "smoke",
            "vectorstore_dir": data_io.resolve_path(args.vectorstore)
            if args.vectorstore
            else (dense if (dense / "chunks.json").exists() else base / "vectorstore"),
            "answers_path": data_io.resolve_path(args.answers)
            if args.answers
            else base / "intents_sandbox.yaml",
        }
    return {
        "model_dir": data_io.resolve_path(args.model_dir)
        if args.model_dir
        else data_io.resolve_path(cfg["paths"]["models_dir"])
        / model_slug(cfg["models"]["primary"]),
        "vectorstore_dir": data_io.resolve_path(args.vectorstore)
        if args.vectorstore
        else data_io.resolve_path(cfg["paths"]["vectorstore_dir"]),
        "answers_path": data_io.resolve_path(args.answers)
        if args.answers
        else data_io.resolve_path(cfg["paths"]["answers"]),
    }


def main(argv: list[str] | None = None) -> int:
    """Titik masuk CLI evaluasi final (butuh frasa konfirmasi persis)."""
    parser = argparse.ArgumentParser(description="Evaluasi FINAL pada test set (terkunci).")
    parser.add_argument(
        "--confirm",
        default=None,
        help=f'Wajib persis: "{CONFIRM_PHRASE}"',
    )
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--vectorstore", default=None)
    parser.add_argument("--answers", default=None)
    parser.add_argument("--method", default="rrf", choices=["dense", "bm25", "rrf"])
    parser.add_argument("--out", default=None, help="Laporan Markdown keluaran.")
    parser.add_argument("--limit", type=int, default=None, help="Batasi baris (uji cepat).")
    parser.add_argument("--sandbox", action="store_true", help="Pakai artefak & test sandbox.")
    args = parser.parse_args(argv)

    if args.confirm != CONFIRM_PHRASE:
        print(
            "Perintah ditolak: evaluasi final butuh konfirmasi eksplisit.\n"
            f'Jalankan: python -m src.final_eval --confirm "{CONFIRM_PHRASE}"'
        )
        return 2

    cfg = data_io.load_config()
    paths = _resolve_paths(cfg, args)
    result = run_final_eval(
        cfg,
        vectorstore_dir=paths["vectorstore_dir"],
        answers_path=paths["answers_path"],
        model_dir=paths["model_dir"],
        method=args.method,
        sandbox=args.sandbox,
        limit=args.limit,
    )

    out_path = (
        data_io.resolve_path(args.out)
        if args.out
        else data_io.resolve_path(cfg["paths"]["reports_dir"])
        / ("sandbox_final_eval.md" if args.sandbox else "final_eval.md")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_report(result), encoding="utf-8")

    from src import experiment_log

    metrics = {
        "accuracy": result["intent_metrics"]["accuracy"],
        "f1_macro": result["intent_metrics"]["f1_macro"],
        "ece": result["ece"],
        "coverage": result["at_tau1"]["coverage"],
    }
    if result["route_metrics"] is not None:
        metrics["route_accuracy"] = result["route_metrics"]["accuracy"]
    n_test = sum(pm["support"] for pm in result["intent_metrics"]["per_class"].values())
    experiment_log.log_experiment(
        fase="Fase 9 - evaluasi final (test)",
        model=Path(result["model_dir"]).name,
        config=(
            f"method={args.method} temperature={result['temperature']:.4f} "
            f"tau1={result['tau1']:.2f} n_test={n_test}"
        ),
        seed=data_io.get_primary_seed(cfg),
        metrics=metrics,
        duration_s=0.0,
        notes="evaluasi final pada test set (sekali)",
        sandbox=args.sandbox,
    )

    print(json.dumps(metrics, indent=2))
    print(f"laporan: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
