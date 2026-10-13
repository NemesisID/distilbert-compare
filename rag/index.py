"""Pembuatan & penyimpanan indeks retrieval (dense + BM25)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from src import data_io

from rag.chunking import Chunk, chunk_pages, chunks_to_records, records_to_chunks
from rag.pdf_extract import extract_dir
from rag.textnorm import get_tokenizer


# ---------------------------------------------------------------------------
# Embedding (sentence-transformers)
# ---------------------------------------------------------------------------


def build_embeddings(
    texts: list[str], model_name: str, max_seq_length: int, batch_size: int
) -> np.ndarray:
    """Encode teks menjadi vektor ternormalisasi (cosine)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    model.max_seq_length = int(max_seq_length)
    emb = model.encode(
        list(texts),
        batch_size=int(batch_size),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(emb, dtype=np.float32)


class Embedder:
    """Pembungkus sentence-transformers untuk encode query."""

    def __init__(self, model_name: str, max_seq_length: int):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = int(max_seq_length)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode daftar teks -> vektor ternormalisasi."""
        emb = self.model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(emb, dtype=np.float32)


# ---------------------------------------------------------------------------
# Indeks
# ---------------------------------------------------------------------------


@dataclass
class RagIndex:
    """Indeks chunk beserta (opsional) embedding dense."""

    chunks: list[Chunk]
    tokenizer_name: str
    embeddings: np.ndarray | None = None
    embedding_model: str | None = None

    def tokenized(self) -> list[list[str]]:
        """Tokenisasi semua chunk untuk BM25."""
        tok = get_tokenizer(self.tokenizer_name)
        return [tok(c.text) for c in self.chunks]

    def bm25(self) -> BM25Okapi:
        """Bangun indeks BM25 dari chunk (dibangun ulang saat dipanggil)."""
        return BM25Okapi(self.tokenized())

    def save(self, out_dir: str | Path) -> Path:
        """Simpan chunks + embedding + meta ke sebuah folder."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "chunks.json").write_text(
            json.dumps(chunks_to_records(self.chunks), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        meta: dict[str, object] = {
            "tokenizer": self.tokenizer_name,
            "embedding_model": self.embedding_model,
            "n_chunks": len(self.chunks),
        }
        if self.embeddings is not None:
            np.save(out / "embeddings.npy", self.embeddings)
            meta["embedding_dim"] = int(self.embeddings.shape[1])
        (out / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return out

    @classmethod
    def load(cls, out_dir: str | Path) -> "RagIndex":
        """Muat indeks dari folder hasil :meth:`save`."""
        out = Path(out_dir)
        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        chunks = records_to_chunks(
            json.loads((out / "chunks.json").read_text(encoding="utf-8"))
        )
        emb = None
        emb_path = out / "embeddings.npy"
        if emb_path.exists():
            emb = np.load(emb_path)
        return cls(
            chunks=chunks,
            tokenizer_name=str(meta["tokenizer"]),
            embeddings=emb,
            embedding_model=meta.get("embedding_model"),
        )


def build_index(
    pdf_dir: str | Path,
    rag_cfg: dict,
    embedding_model: str | None = None,
    with_dense: bool = True,
) -> RagIndex:
    """Bangun indeks dari folder PDF sesuai konfigurasi ``rag``."""
    pages = extract_dir(pdf_dir)
    chunks = chunk_pages(pages, int(rag_cfg["chunk_words"]), int(rag_cfg["chunk_overlap"]))
    embeddings = None
    if with_dense and embedding_model:
        embeddings = build_embeddings(
            [c.text for c in chunks],
            embedding_model,
            int(rag_cfg["embedding_max_seq_length"]),
            int(rag_cfg["embedding_batch_size"]),
        )
    return RagIndex(
        chunks=chunks,
        tokenizer_name=str(rag_cfg["tokenizer"]),
        embeddings=embeddings,
        embedding_model=embedding_model if embeddings is not None else None,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Bangun indeks dari PDF dan simpan ke ``vectorstore``."""
    parser = argparse.ArgumentParser(description="Bangun indeks retrieval PDF.")
    parser.add_argument("--pdf-dir", default=None, help="Folder PDF (default dari config).")
    parser.add_argument("--out", default=None, help="Folder vectorstore keluaran.")
    parser.add_argument("--no-dense", action="store_true", help="Hanya BM25 (tanpa embedding).")
    args = parser.parse_args(argv)

    cfg = data_io.load_config()
    rag_cfg = cfg["rag"]
    pdf_dir = data_io.resolve_path(args.pdf_dir) if args.pdf_dir else data_io.resolve_path(cfg["paths"]["pdf_dir"])
    out_dir = data_io.resolve_path(args.out) if args.out else data_io.resolve_path(cfg["paths"]["vectorstore_dir"])

    idx = build_index(
        pdf_dir,
        rag_cfg,
        embedding_model=cfg["models"]["embedding"],
        with_dense=not args.no_dense,
    )
    saved = idx.save(out_dir)
    print(json.dumps(
        {
            "n_chunks": len(idx.chunks),
            "n_docs": len({c.doc for c in idx.chunks}),
            "dense": idx.embeddings is not None,
            "saved_to": str(saved),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
