"""Kalibrasi (temperature scaling) dan pemilihan ambang tau1.

Hanya memakai data **validasi**. Test set tidak pernah dibaca di sini.
Alur:

1. Hitung logits model pada val.
2. Cari suhu ``T`` yang meminimalkan NLL (temperature scaling).
3. Terapkan ``softmax(logits / T)`` -> probabilitas terkalibrasi.
4. Telusuri ambang ``tau1`` pada grid config; pilih yang memenuhi kriteria
   (akurasi jawaban >= target dan salah-terima OOS <= target), coverage terbesar.
5. Tulis laporan Markdown + ``calibration.json`` di folder model.

Contoh:
    python -m src.calibrate --sandbox --model-dir models/smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src import data_io, experiment_log
from src.data_io import INTENT_COL, SOURCE_COL, TEXT_COL
from src.train import model_slug

SANDBOX_SOURCE = "synthetic"


# ---------------------------------------------------------------------------
# Perhitungan murni (mudah diuji)
# ---------------------------------------------------------------------------


def softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Softmax numerik-stabil dengan pembagian suhu."""
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def temperature_grid(t_min: float, t_max: float, points: int) -> np.ndarray:
    """Grid suhu log-spaced di [t_min, t_max]; selalu menyertakan 1.0."""
    ts = np.geomspace(float(t_min), float(t_max), int(points))
    return np.unique(np.append(ts, 1.0))


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    t_min: float,
    t_max: float,
    points: int,
) -> float:
    """Cari suhu T>0 yang meminimalkan NLL lewat grid log-spaced.

    Grid (bukan optimizer iteratif) dipakai agar selalu stabil/tanpa NaN pada
    data kecil. Batas & jumlah titik berasal dari ``config.yaml``.
    """
    labels = np.asarray(labels)
    rows = np.arange(len(labels))
    best_t, best_nll = 1.0, np.inf
    for t in temperature_grid(t_min, t_max, points):
        p = softmax_with_temperature(logits, t)
        nll = float(-np.log(np.clip(p[rows, labels], 1e-12, 1.0)).mean())
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """ECE: selisih rata-rata keyakinan vs akurasi per bin."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = pred == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(labels)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece) if n else 0.0


def tau_values(tau_min: float, tau_max: float, tau_step: float) -> list[float]:
    """Grid tau1 (dibulatkan 2 desimal agar stabil)."""
    vals = np.arange(float(tau_min), float(tau_max) + 1e-9, float(tau_step))
    return [round(float(v), 2) for v in vals]


def metrics_at_tau(
    probs: np.ndarray, labels: np.ndarray, tau: float, oos_idx: int
) -> dict[str, float]:
    """Metrik bila ambang keyakinan ``tau`` dipakai.

    - ``accepted``: sampel dengan keyakinan (prob maks) >= tau.
    - ``accepted_accuracy``: akurasi pada sampel yang diterima.
    - ``coverage``: proporsi sampel yang diterima.
    - ``oos_false_accept``: proporsi sampel OOS asli yang (keliru) diterima.
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    accepted = conf >= float(tau)
    n = len(labels)
    n_acc = int(accepted.sum())
    acc = float((pred[accepted] == labels[accepted]).mean()) if n_acc else 0.0
    oos_mask = labels == oos_idx
    n_oos = int(oos_mask.sum())
    oos_fa = float((accepted & oos_mask).sum()) / n_oos if n_oos else 0.0
    return {
        "tau": float(tau),
        "coverage": float(n_acc / n) if n else 0.0,
        "accepted_accuracy": acc,
        "n_accepted": n_acc,
        "oos_false_accept": oos_fa,
        "n_oos": n_oos,
    }


def choose_tau1(table: list[dict[str, float]], target_acc: float, target_oos: float) -> tuple[dict, str]:
    """Pilih tau1: coverage terbesar yang memenuhi kedua target.

    Bila tidak ada yang memenuhi, pakai mode ``relaxed``: prioritaskan
    memenuhi batas OOS, lalu akurasi; jika tetap tidak ada, OOS terkecil.
    """
    eligible = [
        r for r in table
        if r["accepted_accuracy"] >= target_acc and r["oos_false_accept"] <= target_oos
    ]
    if eligible:
        return max(eligible, key=lambda r: (r["coverage"], -r["tau"])), "ok"
    within_oos = [r for r in table if r["oos_false_accept"] <= target_oos]
    if within_oos:
        return max(within_oos, key=lambda r: (r["accepted_accuracy"], r["coverage"])), "relaxed"
    return min(table, key=lambda r: r["oos_false_accept"]), "relaxed"


# ---------------------------------------------------------------------------
# Inferensi model
# ---------------------------------------------------------------------------


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_label_map(model_dir: Path) -> tuple[dict[str, int], dict[int, str]]:
    """Baca peta label dari ``label_map.json`` (fallback: config model)."""
    path = Path(model_dir) / "label_map.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        label2id = {str(k): int(v) for k, v in data["label2id"].items()}
        id2label = {int(k): str(v) for k, v in data["id2label"].items()}
        return label2id, id2label
    cfg = data_io.load_config()
    from transformers import AutoConfig

    mc = AutoConfig.from_pretrained(str(model_dir))
    id2label = {int(k): str(v) for k, v in mc.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    return label2id, id2label


def compute_logits(
    model_dir: Path, df, max_length: int, batch_size: int
) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    """Hitung logits model pada DataFrame val (mengembalikan logits & label)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    model.to(device)

    label2id, id2label = load_label_map(model_dir)
    texts = df[TEXT_COL].astype(str).tolist()
    labels = np.asarray([label2id[i] for i in df[INTENT_COL].astype(str)], dtype=np.int64)

    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tokenizer(
                batch, truncation=True, padding="max_length",
                max_length=max_length, return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            chunks.append(out.logits.detach().cpu().numpy())
    logits = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, len(label2id)))
    return logits, labels, id2label


# ---------------------------------------------------------------------------
# Laporan
# ---------------------------------------------------------------------------


def render_report(
    *,
    model_dir: Path,
    val_path: Path,
    temperature: float,
    ece_before: float,
    ece_after: float,
    table: list[dict[str, float]],
    chosen: dict[str, float],
    status: str,
    cal_cfg: dict,
    oos_label: str,
    sandbox: bool,
) -> str:
    """Susun laporan Markdown hasil kalibrasi (val saja)."""
    tag = "[SANDBOX] " if sandbox else ""
    lines = [
        "# Laporan Kalibrasi (temperature scaling) & ambang tau1",
        "",
        f"> **{tag}Data: validasi saja.** Test set TIDAK disentuh.",
        "",
        f"- Folder model: `{model_dir}`",
        f"- Data val: `{val_path}`",
        f"- Suhu terkalibrasi **T = {temperature:.4f}**",
        f"- ECE sebelum kalibrasi: **{ece_before:.4f}**",
        f"- ECE sesudah kalibrasi: **{ece_after:.4f}**",
        "",
        "## Kriteria pemilihan tau1",
        "",
        f"- `target_accepted_accuracy_min` = {cal_cfg['target_accepted_accuracy_min']}",
        f"- `target_oos_false_accept_max` = {cal_cfg['target_oos_false_accept_max']}",
        f"- Intent OOS: `{oos_label}`",
        "",
        f"**Hasil: tau1 = {chosen['tau']:.2f}** (status: `{status}`)",
        "",
        f"- coverage = {chosen['coverage']:.4f}",
        f"- accepted_accuracy = {chosen['accepted_accuracy']:.4f}",
        f"- oos_false_accept = {chosen['oos_false_accept']:.4f}",
        "",
        "## Tabel telusur tau1",
        "",
        "| tau1 | coverage | accepted_accuracy | oos_false_accept | n_accepted |",
        "|------|----------|-------------------|------------------|------------|",
    ]
    for r in table:
        lines.append(
            f"| {r['tau']:.2f} | {r['coverage']:.4f} | {r['accepted_accuracy']:.4f} "
            f"| {r['oos_false_accept']:.4f} | {r['n_accepted']} |"
        )
    lines += [
        "",
        "> Catatan: pemilihan ambang dilakukan HANYA pada val. Test set dipakai",
        "> nanti di `src/final_eval.py` atas perintah eksplisit.",
        "",
    ]
    return "\n".join(lines)


def _load_val(cfg: dict, sandbox: bool, val_arg: str | None):
    """Ambil DataFrame val (atau sandbox val). Tidak pernah menyentuh test."""
    if sandbox:
        path = data_io.resolve_path(cfg["paths"]["sandbox"]).parent / "processed" / "val.csv"
    elif val_arg:
        path = data_io.resolve_path(val_arg)
    else:
        path = data_io.resolve_path(cfg["paths"]["processed_dir"]) / "val.csv"
    df = data_io.read_csv(path, data_io.INTENT_TABLE)
    return df, path


def main(argv: list[str] | None = None) -> int:
    """Titik masuk CLI kalibrasi."""
    parser = argparse.ArgumentParser(description="Kalibrasi suhu & pemilihan tau1 (val).")
    parser.add_argument("--model-dir", default=None, help="Folder model hasil fine-tuning.")
    parser.add_argument("--val", default=None, help="CSV validasi (default dari config).")
    parser.add_argument("--out", default=None, help="Laporan Markdown keluaran.")
    parser.add_argument("--limit", type=int, default=None, help="Batasi jumlah baris val (uji cepat).")
    parser.add_argument("--sandbox", action="store_true", help="Pakai data & laporan sandbox.")
    args = parser.parse_args(argv)

    cfg = data_io.load_config()
    cal_cfg = cfg["calibration"]
    intents = data_io.get_intents(cfg)
    oos_label = str(cfg["routing"]["oos_intent"])
    oos_idx = intents.index(oos_label)

    model_dir = (
        data_io.resolve_path(args.model_dir)
        if args.model_dir
        else data_io.resolve_path(cfg["paths"]["models_dir"]) / model_slug(cfg["models"]["primary"])
    )

    df, val_path = _load_val(cfg, args.sandbox, args.val)
    if args.limit:
        df = df.head(args.limit)

    logits, labels, _ = compute_logits(
        model_dir, df, int(cfg["training"]["max_length"]), int(cal_cfg["batch_size"])
    )

    ece_before = expected_calibration_error(
        softmax_with_temperature(logits, 1.0), labels, int(cal_cfg["n_bins"])
    )
    temperature = fit_temperature(
        logits,
        labels,
        t_min=float(cal_cfg["temperature_min"]),
        t_max=float(cal_cfg["temperature_max"]),
        points=int(cal_cfg["temperature_points"]),
    )
    probs = softmax_with_temperature(logits, temperature)
    ece_after = expected_calibration_error(probs, labels, int(cal_cfg["n_bins"]))

    table = [
        metrics_at_tau(probs, labels, tau, oos_idx)
        for tau in tau_values(cal_cfg["tau1_min"], cal_cfg["tau1_max"], cal_cfg["tau1_step"])
    ]
    chosen, status = choose_tau1(
        table,
        float(cal_cfg["target_accepted_accuracy_min"]),
        float(cal_cfg["target_oos_false_accept_max"]),
    )

    sandbox = args.sandbox or (
        SOURCE_COL in df.columns and (df[SOURCE_COL] == SANDBOX_SOURCE).any()
    )
    out_path = (
        data_io.resolve_path(args.out)
        if args.out
        else data_io.resolve_path(cfg["paths"]["reports_dir"])
        / ("sandbox_calibration_val.md" if sandbox else "calibration_val.md")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_report(
            model_dir=model_dir, val_path=val_path, temperature=temperature,
            ece_before=ece_before, ece_after=ece_after, table=table,
            chosen=chosen, status=status, cal_cfg=cal_cfg, oos_label=oos_label,
            sandbox=sandbox,
        ),
        encoding="utf-8",
    )

    calib_json = model_dir / "calibration.json"
    calib_json.write_text(
        json.dumps(
            {
                "model_dir": str(model_dir),
                "temperature": temperature,
                "tau1": chosen["tau"],
                "selection_status": status,
                "ece_before": ece_before,
                "ece_after": ece_after,
                "oos_intent": oos_label,
                "metrics_at_tau1": chosen,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    experiment_log.log_experiment(
        fase="Fase 4 - kalibrasi (val)",
        model=model_dir.name,
        config=(
            f"temperature={temperature:.4f} tau1={chosen['tau']:.2f} "
            f"status={status} val={val_path.name}"
        ),
        seed=data_io.get_primary_seed(cfg),
        metrics={
            "ece_before": ece_before,
            "ece_after": ece_after,
            "coverage": chosen["coverage"],
            "accepted_accuracy": chosen["accepted_accuracy"],
            "oos_false_accept": chosen["oos_false_accept"],
        },
        duration_s=0.0,
        notes="temperature scaling + pemilihan tau1 pada val",
        sandbox=sandbox,
    )

    print(json.dumps(
        {
            "temperature": temperature,
            "tau1": chosen["tau"],
            "status": status,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "coverage": chosen["coverage"],
            "accepted_accuracy": chosen["accepted_accuracy"],
            "oos_false_accept": chosen["oos_false_accept"],
        },
        indent=2,
    ))
    print(f"laporan: {out_path}")
    print(f"calibration.json: {calib_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
