"""Routing pertanyaan -> jawaban FAQ | kutipan dokumen | fallback.

Menggabungkan:
- klasifikasi intent DistilBERT (probabilitas **terkalibrasi**),
- ambang **tau1** (keyakinan) dari ``models/<slug>/calibration.json``,
- retrieval PDF (dense/BM25/RRF) + ambang **tau2** dari ``vectorstore/tau2.json``,
- jawaban baku FAQ dari ``data/intents.yaml`` (bukan hasil generasi).

Aturan route:
1. intent == OOS  -> ``fallback``
2. keyakinan < tau1 -> ``fallback``
3. intent == tanya_dokumen -> retrieval; bila skor < tau2 -> ``fallback``
   (tidak ditemukan), bila >= tau2 -> ``dokumen`` (dengan kutipan + sumber)
4. intent lain -> ``faq``

Hanya memakai data val/eval. Test set tidak dibaca di sini.

Contoh:
    python -m src.router --sandbox
    python -m src.router --sandbox --ask "berapa kuota peserta setiap angkatan"
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from src import calibrate, data_io, experiment_log
from src.train import model_slug

ROUTE_FAQ = "faq"
ROUTE_DOKUMEN = "dokumen"
ROUTE_FALLBACK = "fallback"

SANDBOX_SOURCE = "synthetic"


# ---------------------------------------------------------------------------
# Logika keputusan murni (mudah diuji)
# ---------------------------------------------------------------------------


def decide_route(
    *,
    intent: str,
    confidence: float,
    oos_intent: str,
    doc_intent: str,
    tau1: float,
    retrieval_score: float | None,
    tau2: float | None,
) -> tuple[str, str]:
    """Tentukan route + alasan dari hasil klasifikasi & retrieval.

    Args:
        retrieval_score: skor kemiripan teratas (dense bila ada, jika tidak BM25),
            atau ``None`` bila retrieval tidak tersedia.
        tau2: ambang skor dokumen; ``None`` berarti tanpa gerbang tau2.

    Returns:
        Pasangan ``(route, alasan)``.
    """
    if intent == oos_intent:
        return ROUTE_FALLBACK, "oos"
    if confidence < float(tau1):
        return ROUTE_FALLBACK, "low_confidence"
    if intent == doc_intent:
        if retrieval_score is None:
            return ROUTE_FALLBACK, "no_retrieval"
        if tau2 is not None and retrieval_score < float(tau2):
            return ROUTE_FALLBACK, "doc_not_found"
        return ROUTE_DOKUMEN, "retrieval_hit"
    return ROUTE_FAQ, "faq_answer"


@dataclass
class RouteDecision:
    """Hasil routing satu pertanyaan."""

    route: str
    intent: str
    confidence: float
    answer: str
    reason: str
    sources: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Pemuat jawaban baku
# ---------------------------------------------------------------------------


def load_answers(path: str | Path) -> dict[str, str]:
    """Baca ``data/intents.yaml`` -> peta intent -> jawaban baku."""
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or []
    answers: dict[str, str] = {}
    for entry in data:
        if not isinstance(entry, dict) or "intent" not in entry:
            continue
        answers[str(entry["intent"])] = str(entry.get("jawaban", "")).strip()
    return answers


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class Router:
    """Muat model + kalibrasi + indeks + jawaban, lalu route pertanyaan."""

    def __init__(
        self,
        cfg: dict,
        *,
        model_dir: str | Path,
        vectorstore_dir: str | Path | None,
        answers_path: str | Path,
        method: str | None = None,
    ):
        self.cfg = cfg
        self.routing = cfg["routing"]
        self.rag_cfg = cfg["rag"]
        self.model_dir = Path(model_dir)
        self.method = method or "rrf"

        self._load_classifier()
        self._load_calibration()
        self.answers = load_answers(answers_path)
        self._load_index(vectorstore_dir)

    # -- pemuatan ---------------------------------------------------------

    def _load_classifier(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.eval()
        self.model.to(self.device)
        self._label2id, self.id2label = calibrate.load_label_map(self.model_dir)

    def _load_calibration(self) -> None:
        cal_path = self.model_dir / "calibration.json"
        if cal_path.exists():
            data = json.loads(cal_path.read_text(encoding="utf-8"))
            self.temperature = float(data["temperature"])
            self.tau1 = float(data["tau1"])
        else:
            warnings.warn(
                f"{cal_path} tidak ada; pakai T=1.0 dan tau1 konservatif "
                f"({self.cfg['calibration']['tau1_max']}).",
                stacklevel=2,
            )
            self.temperature = 1.0
            self.tau1 = float(self.cfg["calibration"]["tau1_max"])

    def _load_index(self, vectorstore_dir: str | Path | None) -> None:
        from rag.index import Embedder, RagIndex

        self.index = None
        self.embedder = None
        self.bm25 = None
        self.tau2 = None
        if vectorstore_dir is None:
            return
        vec = Path(vectorstore_dir)
        if not (vec / "chunks.json").exists():
            warnings.warn(f"Indeks tidak ditemukan di {vec}; route 'dokumen' nonaktif.", stacklevel=2)
            return
        self.index = RagIndex.load(vec)
        self.bm25 = self.index.bm25()
        if self.index.embeddings is not None:
            self.embedder = Embedder(
                self.index.embedding_model or self.cfg["models"]["embedding"],
                int(self.rag_cfg["embedding_max_seq_length"]),
            )
        tau2_path = vec / "tau2.json"
        if tau2_path.exists():
            self.tau2 = float(json.loads(tau2_path.read_text(encoding="utf-8"))["tau2"])

    # -- inferensi --------------------------------------------------------

    def classify(self, text: str) -> tuple[str, float]:
        """Kembalikan ``(intent, keyakinan terkalibrasi)``."""
        enc = self.tokenizer(
            [text],
            truncation=True,
            padding="max_length",
            max_length=int(self.cfg["training"]["max_length"]),
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self.torch.no_grad():
            logits = self.model(**enc).logits.detach().cpu().numpy()
        probs = calibrate.softmax_with_temperature(logits, self.temperature)[0]
        idx = int(probs.argmax())
        return self.id2label[idx], float(probs[idx])

    def retrieve(self, text: str) -> tuple[float | None, list[dict]]:
        """Kembalikan ``(skor_teratas, sumber)`` dari indeks PDF."""
        from rag.retrieve import retrieve

        if self.index is None or not self.index.chunks:
            return None, []
        results = retrieve(
            text,
            self.index,
            embedder=self.embedder,
            method=self.method,
            top_k=int(self.rag_cfg["top_k"]),
            pool=int(self.rag_cfg["rrf_pool"]),
            rrf_k=int(self.rag_cfg["rrf_k"]),
            bm25=self.bm25,
        )
        if not results:
            return None, []
        top = results[0]
        score = top.dense_score if top.dense_score is not None else top.bm25_score
        sources = [
            {
                "doc": r.chunk.doc,
                "page": r.chunk.page,
                "score": float(r.score),
                "dense_score": r.dense_score,
                "bm25_score": r.bm25_score,
                "text": r.chunk.text,
            }
            for r in results
        ]
        return score, sources

    def route(self, text: str) -> RouteDecision:
        """Jalankan routing penuh untuk satu pertanyaan."""
        intent, confidence = self.classify(text)
        score, sources = self.retrieve(text)
        route, reason = decide_route(
            intent=intent,
            confidence=confidence,
            oos_intent=str(self.routing["oos_intent"]),
            doc_intent=str(self.routing["doc_intent"]),
            tau1=self.tau1,
            retrieval_score=score,
            tau2=self.tau2,
        )
        if route == ROUTE_DOKUMEN:
            answer = self._document_answer(sources)
        elif route == ROUTE_FAQ:
            answer = self.answers.get(intent) or str(self.routing["fallback_answer"])
        elif reason == "doc_not_found":
            answer = str(self.routing["not_found_answer"])
        else:
            answer = str(self.routing["fallback_answer"])
        return RouteDecision(
            route=route,
            intent=intent,
            confidence=confidence,
            answer=answer,
            reason=reason,
            sources=sources if route == ROUTE_DOKUMEN else [],
        )

    def _document_answer(self, sources: list[dict]) -> str:
        if not sources:
            return str(self.routing["not_found_answer"])
        top = sources[0]
        return f"{top['text']} (Sumber: {top['doc']}, hal. {top['page']})."


# ---------------------------------------------------------------------------
# Evaluasi end-to-end
# ---------------------------------------------------------------------------


def evaluate(router: Router, rows: list[dict], routes: list[str]) -> dict:
    """Hitung akurasi & metrik per-route untuk data ``mixed_*``."""
    from sklearn.metrics import precision_recall_fscore_support

    expected = [str(r["expected_route"]) for r in rows]
    predicted = []
    details = []
    for row in rows:
        decision = router.route(str(row["question"]))
        predicted.append(decision.route)
        details.append(
            {
                "question": str(row["question"]),
                "expected": str(row["expected_route"]),
                "predicted": decision.route,
                "intent": decision.intent,
                "confidence": decision.confidence,
                "reason": decision.reason,
                "correct": decision.route == str(row["expected_route"]),
            }
        )

    accuracy = (
        sum(d["correct"] for d in details) / len(details) if details else 0.0
    )
    labels = list(dict.fromkeys(routes + expected + predicted))
    precision, recall, f1, support = precision_recall_fscore_support(
        expected, predicted, labels=labels, zero_division=0
    )
    per_route = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }
    confusion = {e: {p: 0 for p in labels} for e in labels}
    for e, p in zip(expected, predicted):
        confusion[e][p] += 1
    return {
        "n": len(details),
        "accuracy": float(accuracy),
        "per_route": per_route,
        "confusion": confusion,
        "details": details,
    }


def render_report(
    *,
    metrics: dict,
    model_dir: Path,
    vectorstore_dir: Path | None,
    answers_path: Path,
    tau1: float,
    tau2: float | None,
    method: str,
    sandbox: bool,
) -> str:
    """Susun laporan Markdown evaluasi routing."""
    tag = "[SANDBOX] " if sandbox else ""
    lines = [
        "# Laporan Evaluasi Routing End-to-End (val)",
        "",
        f"> **{tag}Data: validasi saja.** Test set TIDAK disentuh.",
        "",
        f"- Folder model: `{model_dir}`",
        f"- Vectorstore: `{vectorstore_dir}`",
        f"- Jawaban FAQ: `{answers_path}`",
        f"- Metode retrieval: `{method}`",
        f"- tau1 (keyakinan) = {tau1:.4f}",
        f"- tau2 (dokumen) = {'-' if tau2 is None else f'{tau2:.2f}'}",
        "",
        f"**Akurasi route = {metrics['accuracy']:.4f}** (n = {metrics['n']})",
        "",
        "## Metrik per-route",
        "",
        "| route | precision | recall | f1 | support |",
        "|-------|-----------|--------|----|---------|",
    ]
    for label, m in metrics["per_route"].items():
        lines.append(
            f"| {label} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['support']} |"
        )
    lines += ["", "## Confusion matrix (baris = aktual, kolom = prediksi)", ""]
    labels = list(metrics["confusion"].keys())
    lines.append("| aktual \\ prediksi | " + " | ".join(labels) + " |")
    lines.append("|" + "---|" * (len(labels) + 1))
    for e in labels:
        cells = " | ".join(str(metrics["confusion"][e][p]) for p in labels)
        lines.append(f"| **{e}** | {cells} |")

    wrong = [d for d in metrics["details"] if not d["correct"]]
    lines += ["", f"## Salah route ({len(wrong)} dari {metrics['n']})", ""]
    if wrong:
        lines += [
            "| pertanyaan | aktual | prediksi | intent | keyakinan | alasan |",
            "|------------|--------|----------|--------|-----------|--------|",
        ]
        for d in wrong:
            q = d["question"].replace("|", "\\|")
            lines.append(
                f"| {q} | {d['expected']} | {d['predicted']} | {d['intent']} "
                f"| {d['confidence']:.4f} | {d['reason']} |"
            )
    else:
        lines.append("> Tidak ada kesalahan route.")
    lines += ["", "> Catatan: evaluasi dilakukan pada val; test set menunggu `final_eval.py`.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _sandbox_defaults(cfg: dict) -> dict:
    base = data_io.resolve_path(cfg["paths"]["sandbox"]).parent
    dense = base / "vectorstore_dense"
    vec = dense if (dense / "chunks.json").exists() else (base / "vectorstore")
    return {
        "model_dir": data_io.resolve_path(cfg["paths"]["models_dir"]) / "smoke",
        "vectorstore": vec,
        "answers": base / "intents_sandbox.yaml",
        "e2e": base / "e2e" / "mixed_val.csv",
    }


def main(argv: list[str] | None = None) -> int:
    """Titik masuk CLI routing & evaluasi e2e."""
    parser = argparse.ArgumentParser(description="Routing intent + retrieval (evaluasi val).")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--vectorstore", default=None)
    parser.add_argument("--answers", default=None)
    parser.add_argument("--e2e", default=None, help="CSV mixed (question, expected_route).")
    parser.add_argument("--method", default="rrf", choices=["dense", "bm25", "rrf"])
    parser.add_argument("--out", default=None, help="Laporan Markdown keluaran.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ask", default=None, help="Route satu pertanyaan lalu keluar.")
    parser.add_argument("--tau1", type=float, default=None, help="Override tau1 (diagnostik saja).")
    parser.add_argument("--tau2", type=float, default=None, help="Override tau2 (diagnostik saja).")
    parser.add_argument("--sandbox", action="store_true", help="Pakai model/data sandbox.")
    args = parser.parse_args(argv)

    cfg = data_io.load_config()
    routes = [str(r) for r in cfg["routing"]["routes"]]

    if args.sandbox:
        d = _sandbox_defaults(cfg)
        model_dir = data_io.resolve_path(args.model_dir) if args.model_dir else d["model_dir"]
        vectorstore = data_io.resolve_path(args.vectorstore) if args.vectorstore else d["vectorstore"]
        answers = data_io.resolve_path(args.answers) if args.answers else d["answers"]
        e2e = data_io.resolve_path(args.e2e) if args.e2e else d["e2e"]
    else:
        model_dir = (
            data_io.resolve_path(args.model_dir)
            if args.model_dir
            else data_io.resolve_path(cfg["paths"]["models_dir"]) / model_slug(cfg["models"]["primary"])
        )
        vectorstore = (
            data_io.resolve_path(args.vectorstore)
            if args.vectorstore
            else data_io.resolve_path(cfg["paths"]["vectorstore_dir"])
        )
        answers = data_io.resolve_path(args.answers) if args.answers else data_io.resolve_path(cfg["paths"]["answers"])
        e2e = (
            data_io.resolve_path(args.e2e)
            if args.e2e
            else data_io.resolve_path(cfg["paths"]["e2e_dir"]) / "mixed_val.csv"
        )

    router = Router(
        cfg, model_dir=model_dir, vectorstore_dir=vectorstore, answers_path=answers, method=args.method
    )
    if args.tau1 is not None:
        router.tau1 = float(args.tau1)
    if args.tau2 is not None:
        router.tau2 = float(args.tau2)

    if args.ask:
        print(json.dumps(router.route(args.ask).to_dict(), ensure_ascii=False, indent=2))
        return 0

    df = data_io.read_csv(e2e, data_io.MIXED_TABLE)
    if args.limit:
        df = df.head(args.limit)
    metrics = evaluate(router, df.to_dict("records"), routes)

    out_path = (
        data_io.resolve_path(args.out)
        if args.out
        else data_io.resolve_path(cfg["paths"]["reports_dir"])
        / ("sandbox_e2e_val.md" if args.sandbox else "e2e_val.md")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_report(
            metrics=metrics,
            model_dir=model_dir,
            vectorstore_dir=vectorstore,
            answers_path=answers,
            tau1=router.tau1,
            tau2=router.tau2,
            method=args.method,
            sandbox=args.sandbox,
        ),
        encoding="utf-8",
    )

    experiment_log.log_experiment(
        fase="Fase 6 - routing e2e (val)",
        model=model_dir.name,
        config=(
            f"method={args.method} tau1={router.tau1:.2f} "
            f"tau2={'-' if router.tau2 is None else f'{router.tau2:.2f}'} n={metrics['n']}"
        ),
        seed=data_io.get_primary_seed(cfg),
        metrics={"accuracy": metrics["accuracy"]},
        duration_s=0.0,
        notes="routing intent+retrieval pada mixed_val",
        sandbox=args.sandbox,
    )

    print(json.dumps(
        {
            "accuracy": metrics["accuracy"],
            "n": metrics["n"],
            "per_route_f1": {k: round(v["f1"], 4) for k, v in metrics["per_route"].items()},
        },
        indent=2,
    ))
    print(f"laporan: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
