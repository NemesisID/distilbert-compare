"""Fine-tuning DistilBERT untuk klasifikasi intent.

Disesuaikan dengan **transformers 5.x**: memakai ``eval_strategy``,
``processing_class`` (bukan ``tokenizer``), dan ``warmup_steps`` (karena
``warmup_ratio`` sudah dihapus di transformers 5). ``warmup_ratio`` dari
config tetap dipakai dengan cara dikonversi ke jumlah langkah.

Contoh:
    python -m src.train --sandbox --model cahya/distilbert-base-indonesian \
        --epochs 1 --batch 16 --lr 3e-5 --output models/smoke
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src import data_io, experiment_log
from src.data_io import INTENT_COL, SOURCE_COL, TEXT_COL

SANDBOX_SOURCE = "synthetic"


# ---------------------------------------------------------------------------
# Utilitas
# ---------------------------------------------------------------------------


def make_label_maps(intents: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Buat pemetaan label<->id (deterministik sesuai urutan config)."""
    label2id = {name: i for i, name in enumerate(intents)}
    id2label = {i: name for name, i in label2id.items()}
    return label2id, id2label


def tokenize_frame(df, tokenizer, max_length: int, label2id: dict[str, int]) -> Dataset:
    """Tokenisasi DataFrame menjadi ``datasets.Dataset`` siap latih."""
    enc = tokenizer(
        df[TEXT_COL].astype(str).tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    enc["labels"] = [label2id[i] for i in df[INTENT_COL].astype(str)]
    return Dataset.from_dict(enc)


def _to_array(value) -> np.ndarray:
    """Ubah prediksi/label menjadi ndarray.

    Di transformers 5, ``predictions`` bisa berupa tuple ``(logits, hidden_states,
    ...)`` atau kumpulan array per-batch. Fungsi ini mengambil array logits
    (elemen pertama) dan menggabungkan array per-batch bila perlu.
    """
    if value is None:
        return np.asarray([])
    if isinstance(value, (tuple, list)):
        arrays = [_to_array(v) for v in value]
        if arrays and all(a.ndim == arrays[0].ndim and a.ndim >= 2 for a in arrays):
            try:
                return np.concatenate(arrays, axis=0)
            except ValueError:
                return arrays[0]
        return arrays[0]
    return np.asarray(value)


def compute_metrics(eval_pred, **kwargs) -> dict[str, float]:
    """Hitung accuracy dan macro-F1 untuk Trainer (kompatibel transformers 5.x).

    ``**kwargs`` menampung argumen tambahan (mis. ``compute_result``) yang
    dikirim transformers 5 pada mode metrik per-batch.
    """
    if hasattr(eval_pred, "predictions"):
        preds = _to_array(eval_pred.predictions)
        labels = _to_array(eval_pred.label_ids)
    else:  # fallback untuk bentuk tuple lama
        preds = _to_array(eval_pred[0])
        labels = _to_array(eval_pred[1])

    if preds.ndim > 2:
        preds = preds.reshape(-1, preds.shape[-1])
    predicted = np.argmax(preds, axis=-1).reshape(-1)
    labels = labels.reshape(-1)

    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "f1_macro": float(f1_score(labels, predicted, average="macro", zero_division=0)),
    }


def class_weights_from_labels(labels: list[int], num_labels: int) -> torch.Tensor:
    """Bobot kelas 'balanced': n / (n_kelas * jumlah_kelas_k)."""
    counts = np.bincount(np.asarray(labels), minlength=num_labels).astype(float)
    counts[counts == 0] = 1.0
    weights = len(labels) / (num_labels * counts)
    return torch.tensor(weights, dtype=torch.float)


class WeightedTrainer(Trainer):
    """Trainer dengan opsi bobot kelas untuk data tidak seimbang."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # noqa: ANN001
        """Hitung loss, dengan bobot kelas bila disediakan."""
        if self.class_weights is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


@dataclass
class TrainResult:
    """Hasil satu kali pelatihan."""

    model_name: str
    seed: int
    lr: float
    batch: int
    epochs: int
    metrics: dict[str, float]
    duration_s: float
    output_dir: str


def warmup_steps_for(ratio: float, n_train: int, batch: int, epochs: int) -> int:
    """Konversi warmup_ratio -> warmup_steps (transformers 5 tak punya ratio)."""
    steps_per_epoch = max(1, int(np.ceil(n_train / max(1, batch))))
    return max(0, int(round(ratio * steps_per_epoch * epochs)))


# ---------------------------------------------------------------------------
# Pelatihan
# ---------------------------------------------------------------------------


def train_single(
    *,
    model_name: str,
    train_df,
    val_df,
    intents: list[str],
    seed: int,
    lr: float,
    batch: int,
    epochs: int,
    output_dir: str | Path,
    max_length: int,
    weight_decay: float,
    warmup_ratio: float,
    patience: int,
    metric_for_best: str,
    use_class_weight: bool,
    num_threads: int,
) -> TrainResult:
    """Latih satu model dengan satu konfigurasi dan simpan yang terbaik."""
    torch.set_num_threads(num_threads)
    set_seed(seed)

    label2id, id2label = make_label_maps(intents)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Catatan: tokenizer uncased (cahya) otomatis lowercase; cased tidak dipaksa.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(intents),
        id2label=id2label,
        label2id=label2id,
    )

    train_ds = tokenize_frame(train_df, tokenizer, max_length, label2id)
    val_ds = tokenize_frame(val_df, tokenizer, max_length, label2id)

    weights = None
    if use_class_weight:
        weights = class_weights_from_labels(train_ds["labels"], len(intents))

    cuda = torch.cuda.is_available()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        learning_rate=lr,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps_for(warmup_ratio, len(train_df), batch, epochs),
        eval_strategy="epoch",          # transformers 5.x
        save_strategy="epoch",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best,
        greater_is_better=True,
        save_total_limit=1,
        save_only_model=True,           # jangan simpan optimizer/scheduler (hemat disk)
        report_to="none",
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,       # WAJIB 0 di Windows
        fp16=cuda,                      # fp16 hanya bila ada GPU
        use_cpu=not cuda,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,     # transformers 5.x (bukan tokenizer=)
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
        class_weights=weights,
    )

    start = time.perf_counter()
    trainer.train()
    duration = time.perf_counter() - start

    metrics = trainer.evaluate()
    final = {
        "accuracy": float(metrics.get("eval_accuracy", float("nan"))),
        "f1_macro": float(metrics.get("eval_f1_macro", float("nan"))),
        "eval_loss": float(metrics.get("eval_loss", float("nan"))),
        "best_metric": float(trainer.state.best_metric)
        if trainer.state.best_metric is not None
        else float("nan"),
    }

    # Simpan model + tokenizer + peta label
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    (out / "label_map.json").write_text(
        json.dumps({"label2id": label2id, "id2label": id2label}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "train_meta.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "seed": seed,
                "lr": lr,
                "batch": batch,
                "epochs": epochs,
                "max_length": max_length,
                "use_class_weight": use_class_weight,
                "lowercase_tokenizer": bool(getattr(tokenizer, "do_lower_case", False)),
                "duration_s": duration,
                "metrics": final,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return TrainResult(
        model_name=model_name,
        seed=seed,
        lr=lr,
        batch=batch,
        epochs=epochs,
        metrics=final,
        duration_s=duration,
        output_dir=str(out),
    )


# ---------------------------------------------------------------------------
# Pencarian hyperparameter & pengulangan seed
# ---------------------------------------------------------------------------


def model_slug(model_name: str) -> str:
    """Nama folder aman dari nama model HF."""
    return model_name.split("/")[-1]


def _common_kwargs(tcfg: dict, intents, train_df, val_df, use_class_weight: bool) -> dict:
    """Kumpulan argumen umum untuk ``train_single`` dari config."""
    return dict(
        train_df=train_df,
        val_df=val_df,
        intents=intents,
        max_length=int(tcfg["max_length"]),
        weight_decay=float(tcfg["weight_decay"]),
        warmup_ratio=float(tcfg["warmup_ratio"]),
        patience=int(tcfg["early_stopping_patience"]),
        metric_for_best=str(tcfg["metric_for_best_model"]),
        use_class_weight=use_class_weight,
        num_threads=int(tcfg["torch_num_threads"]),
    )


def _log(fase: str, model_name: str, cfg_text: str, seed: int, result: TrainResult, sandbox: bool, notes: str) -> None:
    """Catat satu run ke EXPERIMENTS.md."""
    experiment_log.log_experiment(
        fase=fase,
        model=model_name,
        config=cfg_text,
        seed=seed,
        metrics=result.metrics,
        duration_s=result.duration_s,
        notes=notes,
        sandbox=sandbox,
    )


def run_search(
    *,
    model_name: str,
    train_df,
    val_df,
    intents,
    tcfg: dict,
    seed: int,
    epochs: int,
    runs_root: Path,
    sandbox: bool,
    use_class_weight: bool,
) -> list[TrainResult]:
    """Telusuri grid lr x batch pada data validasi (bukan test).

    Mengembalikan daftar hasil; tiap konfigurasi disimpan di
    ``runs_root/search/lr{lr}_bs{batch}/``.
    """
    results: list[TrainResult] = []
    common = _common_kwargs(tcfg, intents, train_df, val_df, use_class_weight)
    for lr in tcfg["learning_rates"]:
        for batch in tcfg["batch_sizes"]:
            run_dir = runs_root / "search" / f"lr{lr}_bs{batch}" / f"seed{seed}"
            res = train_single(
                model_name=model_name,
                seed=seed,
                lr=float(lr),
                batch=int(batch),
                epochs=epochs,
                output_dir=run_dir,
                **common,
            )
            results.append(res)
            _log(
                "Fase 3 - search (val)",
                model_name,
                f"lr={lr} batch={batch} epochs={epochs} class_weight={use_class_weight}",
                seed,
                res,
                sandbox,
                "grid search pada val",
            )
    return results


def run_multiseed(
    *,
    model_name: str,
    train_df,
    val_df,
    intents,
    tcfg: dict,
    lr: float,
    batch: int,
    seeds: list[int],
    epochs: int,
    runs_root: Path,
    sandbox: bool,
    use_class_weight: bool,
) -> list[TrainResult]:
    """Ulangi konfigurasi terbaik dengan beberapa seed."""
    results: list[TrainResult] = []
    common = _common_kwargs(tcfg, intents, train_df, val_df, use_class_weight)
    for seed in seeds:
        run_dir = runs_root / "multiseed" / f"lr{lr}_bs{batch}" / f"seed{seed}"
        res = train_single(
            model_name=model_name,
            seed=int(seed),
            lr=float(lr),
            batch=int(batch),
            epochs=epochs,
            output_dir=run_dir,
            **common,
        )
        results.append(res)
        _log(
            "Fase 3 - multiseed",
            model_name,
            f"lr={lr} batch={batch} epochs={epochs} class_weight={use_class_weight}",
            int(seed),
            res,
            sandbox,
            "pengulangan seed",
        )
    return results


def aggregate(results: list[TrainResult]) -> dict[str, float]:
    """Rata-rata dan simpangan baku metrik utama dari beberapa run."""
    if not results:
        return {}
    acc = np.asarray([r.metrics.get("accuracy", np.nan) for r in results], dtype=float)
    f1 = np.asarray([r.metrics.get("f1_macro", np.nan) for r in results], dtype=float)
    return {
        "accuracy_mean": float(np.nanmean(acc)),
        "accuracy_std": float(np.nanstd(acc)),
        "f1_macro_mean": float(np.nanmean(f1)),
        "f1_macro_std": float(np.nanstd(f1)),
        "n_runs": len(results),
    }


def publish_best(best: TrainResult, publish_dir: Path) -> None:
    """Salin berkas model terbaik ke folder kanonik ``publish_dir``."""
    publish_dir.mkdir(parents=True, exist_ok=True)
    src = Path(best.output_dir)
    keep = ("config.json", "model.safetensors", "label_map.json",
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt")
    for name in keep:
        s = src / name
        if s.exists():
            shutil.copy2(s, publish_dir / name)
    (publish_dir / "selected.json").write_text(
        json.dumps(
            {
                "model_name": best.model_name,
                "seed": best.seed,
                "lr": best.lr,
                "batch": best.batch,
                "epochs": best.epochs,
                "metrics": best.metrics,
                "source_run": best.output_dir,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_frames(cfg: dict, args: argparse.Namespace):
    if args.sandbox:
        base = data_io.resolve_path(cfg["paths"]["sandbox"]).parent
        train_path = base / "augmented" / "train_aug.csv"
        val_path = base / "processed" / "val.csv"
    else:
        train_path = (
            data_io.resolve_path(args.train)
            if args.train
            else data_io.resolve_path(cfg["paths"]["augmented_dir"]) / "train_aug.csv"
        )
        val_path = (
            data_io.resolve_path(args.val)
            if args.val
            else data_io.resolve_path(cfg["paths"]["processed_dir"]) / "val.csv"
        )
    train_df = data_io.read_csv(train_path, data_io.INTENT_TABLE)
    val_df = data_io.read_csv(val_path, data_io.INTENT_TABLE)
    if args.limit_train:
        train_df = train_df.head(args.limit_train)
    if args.limit_val:
        val_df = val_df.head(args.limit_val)
    sandbox = SOURCE_COL in train_df.columns and (train_df[SOURCE_COL] == SANDBOX_SOURCE).any()
    return train_df, val_df, train_path, val_path, sandbox


def main(argv: list[str] | None = None) -> int:
    """Titik masuk CLI."""
    parser = argparse.ArgumentParser(description="Fine-tuning DistilBERT.")
    parser.add_argument(
        "--mode",
        choices=["single", "search", "multiseed"],
        default="single",
        help="single: satu run; search: grid lr x batch di val; multiseed: ulangi konfigurasi terbaik.",
    )
    parser.add_argument("--model", default=None, help="Nama model HF.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--class-weight", action="store_true")
    parser.add_argument("--train", default=None)
    parser.add_argument("--val", default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--sandbox", action="store_true")
    args = parser.parse_args(argv)

    cfg = data_io.load_config()
    tcfg = cfg["training"]
    intents = data_io.get_intents(cfg)

    model_name = args.model or cfg["models"]["primary"]
    seed = int(args.seed if args.seed is not None else data_io.get_primary_seed(cfg))
    lr = float(args.lr if args.lr is not None else tcfg["learning_rates"][0])
    batch = int(args.batch if args.batch is not None else tcfg["batch_sizes"][0])
    epochs = int(args.epochs if args.epochs is not None else tcfg["epochs_max"])

    train_df, val_df, train_path, val_path, sandbox = _load_frames(cfg, args)
    common = _common_kwargs(tcfg, intents, train_df, val_df, bool(args.class_weight))

    # Folder kanonik model + folder run mentah.
    publish_dir = (
        data_io.resolve_path(args.output)
        if args.output
        else data_io.resolve_path(cfg["paths"]["models_dir"]) / model_slug(model_name)
    )
    runs_root = publish_dir / "runs"

    if args.mode == "single":
        result = train_single(
            model_name=model_name,
            seed=seed,
            lr=lr,
            batch=batch,
            epochs=epochs,
            output_dir=publish_dir,
            **common,
        )
        print(json.dumps(result.metrics, indent=2))
        print(f"model disimpan di: {result.output_dir}")
        _log(
            "Fase 3 - fine-tuning",
            model_name,
            f"lr={lr} batch={batch} epochs={epochs} max_len={tcfg['max_length']} "
            f"class_weight={args.class_weight} train={train_path.name}",
            seed,
            result,
            sandbox,
            "fine-tuning DistilBERT (single)",
        )
        return 0

    if args.mode == "search":
        results = run_search(
            model_name=model_name,
            train_df=train_df,
            val_df=val_df,
            intents=intents,
            tcfg=tcfg,
            seed=seed,
            epochs=epochs,
            runs_root=runs_root,
            sandbox=sandbox,
            use_class_weight=bool(args.class_weight),
        )
        best = max(results, key=lambda r: r.metrics.get("f1_macro", float("-inf")))
        publish_best(best, publish_dir)
        print(json.dumps({"best": best.__dict__, "all": [r.metrics for r in results]}, indent=2, ensure_ascii=False))
        print(f"konfigurasi terbaik: lr={best.lr} batch={best.batch} f1_macro={best.metrics['f1_macro']:.4f}")
        return 0

    # mode == "multiseed"
    results = run_multiseed(
        model_name=model_name,
        train_df=train_df,
        val_df=val_df,
        intents=intents,
        tcfg=tcfg,
        lr=lr,
        batch=batch,
        seeds=data_io.get_seeds(cfg),
        epochs=epochs,
        runs_root=runs_root,
        sandbox=sandbox,
        use_class_weight=bool(args.class_weight),
    )
    best = max(results, key=lambda r: r.metrics.get("f1_macro", float("-inf")))
    publish_best(best, publish_dir)
    agg = aggregate(results)
    print(json.dumps(agg, indent=2))
    print(f"model terbaik dari seed {best.seed} disalin ke: {publish_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
