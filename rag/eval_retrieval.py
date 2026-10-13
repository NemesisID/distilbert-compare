"""Evaluasi retrieval pada data ``rag_eval`` + pemilihan ambang tau2.

Hanya memakai data validasi/evaluasi retrieval (``qa_pairs``, ``in_doc``,
``out_doc``). Test set klasifikasi tidak pernah dibaca di sini.

Alur:

1. Muat indeks (dense + BM25) dari ``vectorstore``.
2. Ukur retrieval ``qa_pairs``: recall@k (doc+page emas) dan MRR.
3. Ukur skor kemiripan dense top-1 untuk ``in_doc`` dan ``out_doc``.
4. Telusuri ambang ``tau2`` pada grid config; pilih yang memenuhi kriteria
   (in-doc terjawab >= target dan out-doc salah-terima <= target).
5. Tulis laporan Markdown.

Contoh:
    python -m rag.eval_retrieval --sandbox --method bm25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src import data_io, experiment_log

from rag.index import Embedder, RagIndex
from rag.retrieve import Retrieved, retrieve


# ---------------------------------------------------------------------------
# Perhitungan murni (mudah diuji)
# ---------------------------------------------------------------------------


def hit_at_k(results: list[Retrieved], doc: str, page: int, k: int) -> bool:
    """True bila doc+page emas muncul di ``k`` hasil teratas."""
    return any(
        r.chunk.doc == doc and r.chunk.page == int(page) for r in results[: int(k)]
    )


def reciprocal_rank(results: list[Retrieved], doc: str, page: int) -> float:
    """1/rank kemunculan pertama doc+page emas (0 bila tidak ada)."""
    for rank, r in enumerate(results, start=1):
        if r.chunk.doc == doc and r.chunk.page == int(page):
            return 1.0 / rank
    return 0.0


def evaluate_qa(
    rows,
    index: RagIndex,
    *,
    embedder: Embedder | None,
    method: str,
    top_k: int,
    pool: int,
    rrf_k: int,
    bm25=None,
) -> dict:
    """Hitung recall@1..k dan MRR untuk tabel ``qa_pairs``."""
    recalls: dict[int, list[float]] = {k: [] for k in range(1, int(top_k) + 1)}
    rr: list[float] = []
    for row in rows:
        res = retrieve(
            str(row["question"]),
            index,
            embedder=embedder,
            method=method,
            top_k=int(top_k),
            pool=int(pool),
            rrf_k=int(rrf_k),
            bm25=bm25,
        )
        doc, page = str(row["doc"]), int(row["page"])
        for k in recalls:
            recalls[k].append(1.0 if hit_at_k(res, doc, page, k) else 0.0)
        rr.append(reciprocal_rank(res, doc, page))
    return {
        "n": len(rows),
        "recall_at_k": {int(k): float(np.mean(v)) for k, v in recalls.items()},
        "mrr": float(np.mean(rr)) if rr else 0.0,
    }


def top1_similarity(query: str, index: RagIndex, embedder: Embedder) -> float:
    """Kemiripan cosine top-1 sebuah query terhadap seluruh chunk."""
    if index.embeddings is None:
        raise ValueError("Indeks tidak punya embedding dense.")
    q = embedder.encode([query])[0]
    return float((index.embeddings @ q).max())


def tau2_values(tau_min: float, tau_max: float, tau_step: float) -> list[float]:
    """Grid tau2 (dibulatkan 2 desimal agar stabil)."""
    vals = np.arange(float(tau_min), float(tau_max) + 1e-9, float(tau_step))
    return [round(float(v), 2) for v in vals]


def metrics_at_tau2(in_scores: list[float], out_scores: list[float], tau: float) -> dict:
    """Metrik bila ambang kemiripan ``tau`` dipakai untuk memutuskan retrieval.

    - ``in_doc_recall``: proporsi in-doc dengan skor top-1 >= tau (terjawab).
    - ``out_doc_false_accept``: proporsi out-doc dengan skor top-1 >= tau (keliru diterima).
    """
    ins = np.asarray(in_scores, dtype=float)
    outs = np.asarray(out_scores, dtype=float)
    return {
        "tau": float(tau),
        "in_doc_recall": float((ins >= float(tau)).mean()) if ins.size else 0.0,
        "out_doc_false_accept": float((outs >= float(tau)).mean()) if outs.size else 0.0,
        "n_in_doc": int(ins.size),
        "n_out_doc": int(outs.size),
    }


def select_tau2(table: list[dict], recall_min: float, false_accept_max: float) -> tuple[dict, str]:
    """Pilih tau2 terkecil yang memenuhi kedua target (coverage terbesar).

    Bila tak ada yang memenuhi, mode ``relaxed``: dahulukan batas false-accept,
    lalu recall tertinggi; jika tetap tidak ada, false-accept terkecil.
    """
    eligible = [
        r for r in table
        if r["in_doc_recall"] >= recall_min and r["out_doc_false_accept"] <= false_accept_max
    ]
    if eligible:
        return min(eligible, key=lambda r: r["tau"]), "ok"
    within_fa = [r for r in table if r["out_doc_false_accept"] <= false_accept_max]
    if within_fa:
        return max(within_fa, key=lambda r: (r["in_doc_recall"], -r["tau"])), "relaxed"
    return min(table, key=lambda r: r["out_doc_false_accept"]), "relaxed"


# ---------------------------------------------------------------------------
# Pemuat data eval
# ---------------------------------------------------------------------------


def _questions(path: Path) -> list[str]:
    df = data_io.read_csv(path, data_io.QUESTION_TABLE)
    return [str(q) for q in df["question"]]


def _qa_rows(path: Path):
    return data_io.read_csv(path, data_io.QA_PAIRS_TABLE)


# ---------------------------------------------------------------------------
# Laporan
# ---------------------------------------------------------------------------


def render_report(
    *,
    method: str,
    index: RagIndex,
    qa_metrics: dict,
    table: list[dict] | None,
    chosen: dict | None,
    status: str,
    rag_cfg: dict,
    sandbox: bool,
) -> str:
    """Susun laporan Markdown hasil evaluasi retrieval (val/eval saja)."""
    tag = "[SANDBOX] " if sandbox else ""
    lines = [
        "# Laporan Evaluasi Retrieval PDF & ambang tau2",
        "",
        f"> **{tag}Data: evaluasi retrieval saja.** Test set klasifikasi TIDAK disentuh.",
        "",
        f"- Metode retrieval: `{method}`",
        f"- Jumlah chunk: {len(index.chunks)}",
        f"- Jumlah dokumen: {len({c.doc for c in index.chunks})}",
        f"- Dense aktif: {index.embeddings is not None}",
        f"- Model embedding: `{index.embedding_model}`",
        "",
        "## Metrik qa_pairs (doc + halaman emas)",
        "",
        f"- n = {qa_metrics['n']}",
        f"- MRR = {qa_metrics['mrr']:.4f}",
        "",
        "| k | recall@k |",
        "|---|----------|",
    ]
    for k, v in qa_metrics["recall_at_k"].items():
        lines.append(f"| {k} | {v:.4f} |")

    if table is None or chosen is None:
        lines += [
            "",
            "## Pemilihan tau2",
            "",
            "> Dilewati: butuh embedding dense. Jalankan ulang tanpa `--no-dense`.",
            "",
        ]
        return "\n".join(lines)

    lines += [
        "",
        "## Kriteria pemilihan tau2",
        "",
        f"- `tau2_accept_recall_min` = {rag_cfg['tau2_accept_recall_min']}",
        f"- `tau2_false_accept_max` = {rag_cfg['tau2_false_accept_max']}",
        "",
        f"**Hasil: tau2 = {chosen['tau']:.2f}** (status: `{status}`)",
        "",
        f"- in_doc_recall = {chosen['in_doc_recall']:.4f} (n={chosen['n_in_doc']})",
        f"- out_doc_false_accept = {chosen['out_doc_false_accept']:.4f} (n={chosen['n_out_doc']})",
        "",
        "## Tabel telusur tau2",
        "",
        "| tau2 | in_doc_recall | out_doc_false_accept |",
        "|------|---------------|----------------------|",
    ]
    for r in table:
        lines.append(
            f"| {r['tau']:.2f} | {r['in_doc_recall']:.4f} | {r['out_doc_false_accept']:.4f} |"
        )
    lines += [
        "",
        "> Catatan: tau2 ditentukan dari data eval retrieval, bukan test set.",
        "",
    ]
    lowest = min(table, key=lambda r: r["tau"]) if table else None
    if lowest is not None and lowest["in_doc_recall"] < float(rag_cfg["tau2_accept_recall_min"]):
        lines += [
            "> **Peringatan skala:** ambang tau2 terendah pada grid "
            f"({lowest['tau']:.2f}) pun belum mencapai target in-doc recall "
            f"({lowest['in_doc_recall']:.4f} < {rag_cfg['tau2_accept_recall_min']}). "
            "Model embedding `distiluse-base-multilingual-cased-v2` menurunkan "
            "kemiripan saat jawaban tertanam dalam chunk panjang. Perlu menurunkan "
            "`rag.tau2_min` (mis. ke 0.05) atau memakai skor metode lain; "
            "keputusan ada pada pemilik proyek.",
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _sandbox_eval_dir(cfg: dict) -> Path:
    """Folder data eval retrieval sandbox (``data/sandbox/rag_eval``)."""
    return data_io.resolve_path(cfg["paths"]["sandbox"]).parent / "rag_eval"


def _sandbox_vectorstore_dir(cfg: dict) -> Path:
    """Folder indeks sandbox (``data/sandbox/vectorstore``)."""
    return data_io.resolve_path(cfg["paths"]["sandbox"]).parent / "vectorstore"


def main(argv: list[str] | None = None) -> int:
    """Titik masuk CLI evaluasi retrieval."""
    parser = argparse.ArgumentParser(description="Evaluasi retrieval PDF & pemilihan tau2.")
    parser.add_argument("--vectorstore", default=None, help="Folder indeks (default config).")
    parser.add_argument("--rag-eval-dir", default=None, help="Folder data eval (default config).")
    parser.add_argument("--method", default="rrf", choices=["dense", "bm25", "rrf"])
    parser.add_argument("--out", default=None, help="Laporan Markdown keluaran.")
    parser.add_argument("--limit", type=int, default=None, help="Batasi baris qa_pairs (uji cepat).")
    parser.add_argument("--sandbox", action="store_true", help="Pakai data & laporan sandbox.")
    args = parser.parse_args(argv)

    cfg = data_io.load_config()
    rag_cfg = cfg["rag"]
    sandbox = bool(args.sandbox)

    if sandbox:
        vec_dir = data_io.resolve_path(args.vectorstore) if args.vectorstore else _sandbox_vectorstore_dir(cfg)
        eval_dir = data_io.resolve_path(args.rag_eval_dir) if args.rag_eval_dir else _sandbox_eval_dir(cfg)
    else:
        vec_dir = (
            data_io.resolve_path(args.vectorstore)
            if args.vectorstore
            else data_io.resolve_path(cfg["paths"]["vectorstore_dir"])
        )
        eval_dir = (
            data_io.resolve_path(args.rag_eval_dir)
            if args.rag_eval_dir
            else data_io.resolve_path(cfg["paths"]["rag_eval_dir"])
        )

    index = RagIndex.load(vec_dir)
    embedder: Embedder | None = None
    if index.embeddings is not None:
        embedder = Embedder(
            index.embedding_model or cfg["models"]["embedding"],
            int(rag_cfg["embedding_max_seq_length"]),
        )

    top_k = int(rag_cfg["top_k"])
    pool = int(rag_cfg["rrf_pool"])
    rrf_k = int(rag_cfg["rrf_k"])
    bm25 = index.bm25()

    qa_rows = _qa_rows(eval_dir / "qa_pairs.csv")
    if args.limit:
        qa_rows = qa_rows.head(args.limit)
    qa_metrics = evaluate_qa(
        qa_rows.to_dict("records"),
        index,
        embedder=embedder,
        method=args.method,
        top_k=top_k,
        pool=pool,
        rrf_k=rrf_k,
        bm25=bm25,
    )

    table = None
    chosen = None
    status = "skipped"
    if embedder is not None:
        in_scores = [top1_similarity(q, index, embedder) for q in _questions(eval_dir / "in_doc.csv")]
        out_scores = [top1_similarity(q, index, embedder) for q in _questions(eval_dir / "out_doc.csv")]
        table = [
            metrics_at_tau2(in_scores, out_scores, t)
            for t in tau2_values(rag_cfg["tau2_min"], rag_cfg["tau2_max"], rag_cfg["tau2_step"])
        ]
        chosen, status = select_tau2(
            table,
            float(rag_cfg["tau2_accept_recall_min"]),
            float(rag_cfg["tau2_false_accept_max"]),
        )

    out_path = (
        data_io.resolve_path(args.out)
        if args.out
        else data_io.resolve_path(cfg["paths"]["reports_dir"])
        / ("sandbox_rag_eval.md" if sandbox else "rag_eval.md")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_report(
            method=args.method,
            index=index,
            qa_metrics=qa_metrics,
            table=table,
            chosen=chosen,
            status=status,
            rag_cfg=rag_cfg,
            sandbox=sandbox,
        ),
        encoding="utf-8",
    )

    if chosen is not None:
        idx_json = vec_dir / "tau2.json"
        idx_json.write_text(
            json.dumps(
                {
                    "method": args.method,
                    "tau2": chosen["tau"],
                    "selection_status": status,
                    "metrics_at_tau2": chosen,
                    "qa_metrics": qa_metrics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    experiment_log.log_experiment(
        fase="Fase 5 - retrieval PDF (eval)",
        model=args.method,
        config=(
            f"method={args.method} top_k={top_k} rrf_k={rrf_k} "
            f"tau2={(chosen or {}).get('tau')} status={status}"
        ),
        seed=data_io.get_primary_seed(cfg),
        metrics={
            "mrr": qa_metrics["mrr"],
            "recall_at_1": qa_metrics["recall_at_k"].get(1, 0.0),
            "recall_at_k": qa_metrics["recall_at_k"].get(top_k, 0.0),
            **(chosen or {}),
        },
        duration_s=0.0,
        notes="evaluasi retrieval PDF pada data rag_eval",
        sandbox=sandbox,
    )

    summary = {
        "method": args.method,
        "n_chunks": len(index.chunks),
        "dense": index.embeddings is not None,
        "mrr": qa_metrics["mrr"],
        "recall_at_k": qa_metrics["recall_at_k"],
        "tau2": (chosen or {}).get("tau"),
        "status": status,
    }
    print(json.dumps(summary, indent=2))
    print(f"laporan: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
