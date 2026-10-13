"""Bangkitkan data SANDBOX sintetis untuk pipeline PDF/retrieval (Fase 5).

Menulis:
- ``data/sandbox/pdf/*.pdf``          : PDF sintetis berteks (2 dokumen, 2 halaman).
- ``data/sandbox/rag_eval/qa_pairs.csv`` : pertanyaan + doc + halaman emas.
- ``data/sandbox/rag_eval/in_doc.csv``   : pertanyaan yang ADA di dokumen.
- ``data/sandbox/rag_eval/out_doc.csv``  : pertanyaan yang TIDAK ADA di dokumen.

SEMUA isi di sini SINTETIS (``source=synthetic``). Jangan pernah dilaporkan
sebagai hasil penelitian atau dicampur dengan data nyata.

Contoh:
    .venv\\Scripts\\python.exe -m scripts.make_sandbox_rag
"""

from __future__ import annotations

import csv
from pathlib import Path

import pymupdf

from src import data_io

# ---------------------------------------------------------------------------
# Teks sintetis per halaman (fiktif; bukan data mitra)
# ---------------------------------------------------------------------------

PAGES: dict[str, dict[int, list[str]]] = {
    "panduan_pendaftaran.pdf": {
        1: [
            "PENDAFTARAN PROGRAM INKUBATOR (DOKUMEN CONTOH SINTETIS)",
            "Syarat pendaftaran meliputi kartu tanda mahasiswa aktif, proposal usaha",
            "ringkas tiga halaman, dan surat izin dari ketua program studi.",
            "Berkas diserahkan dalam format PDF melalui formulir daring sebelum",
            "batas waktu pendaftaran yang diumumkan pengelola setiap awal semester.",
            "Pendaftar wajib mengisi data identitas tim beserta pembagian peran.",
            "Kuota peserta setiap angkatan dibatasi dua puluh tim usaha aktif.",
            "Berkas yang tidak lengkap akan dikembalikan untuk dilengkapi.",
            "Pengelola tidak memungut biaya pendaftaran dalam bentuk apa pun.",
            "Hanya tim dengan anggota aktif yang dapat melanjutkan ke tahap berikutnya.",
        ],
        2: [
            "ALUR SELEKSI DAN VERIFIKASI (DOKUMEN CONTOH SINTETIS)",
            "Seleksi administrasi memeriksa kelengkapan berkas pendaftaran.",
            "Seleksi substansi menilai kelayakan gagasan melalui presentasi daring.",
            "Pengumuman hasil seleksi disampaikan lewat surat elektronik peserta.",
            "Tim yang lolos diundang mengikuti pembekalan awal inkubasi.",
            "Penilaian substansi mempertimbangkan keunikan gagasan dan potensi pasar.",
            "Keputusan tim penilai bersifat final dan tidak dapat diganggu gugat.",
            "Peserta yang tidak lolos dapat mengajukan kembali pada periode berikutnya.",
            "Verifikasi data tim dilakukan sebelum pengumuman resmi dikeluarkan.",
        ],
    },
    "jadwal_program.pdf": {
        1: [
            "JADWAL KEGIATAN INKUBASI (DOKUMEN CONTOH SINTETIS)",
            "Pembekalan awal dilaksanakan pada pekan pertama setiap angkatan.",
            "Pendampingan usaha berjalan dua kali setiap bulan selama enam bulan.",
            "Klinik keuangan dijadwalkan pada pertengahan periode inkubasi.",
            "Evaluasi tengah periode dilakukan bersama mentor pendamping.",
            "Penutupan program ditandai dengan gelaran pameran produk.",
            "Bimbingan teknis pemasaran diberikan menjelang pameran produk akhir.",
            "Peserta wajib menghadiri minimal delapan puluh persen jadwal kegiatan.",
            "Perubahan jadwal diinformasikan paling lambat tiga hari sebelumnya.",
        ],
        2: [
            "KETENTUAN LUARAN PROGRAM (DOKUMEN CONTOH SINTETIS)",
            "Setiap tim wajib menyusun laporan kemajuan bulanan secara tertulis.",
            "Luaran akhir berupa prototipe produk dan rencana bisnis satu tahun.",
            "Tim menyerahkan berkas pengeluaran keuangan yang telah divalidasi.",
            "Revisi berkas diperbolehkan paling lambat satu pekan sebelum penutupan.",
            "Pelanggaran ketentuan luaran menyebabkan penundaan kelulusan program.",
            "Laporan akhir diseminarkan di hadapan penguji internal inkubator.",
            "Dokumen luaran disimpan pengelola sebagai arsip capaian program.",
        ],
    },
}

# question, doc, page
QA_PAIRS: list[tuple[str, str, int]] = [
    ("apa saja syarat pendaftaran program inkubator", "panduan_pendaftaran.pdf", 1),
    ("bagaimana alur seleksi dan verifikasi berkas", "panduan_pendaftaran.pdf", 2),
    ("kapan jadwal pendampingan usaha dilaksanakan", "jadwal_program.pdf", 1),
    ("apa ketentuan luaran akhir program inkubasi", "jadwal_program.pdf", 2),
]

IN_DOC: list[str] = [
    "berapa kuota peserta setiap angkatan",
    "bagaimana cara pengumuman hasil seleksi",
    "kapan klinik keuangan dijadwalkan",
    "apa sanksi bila melanggar ketentuan luaran",
]

OUT_DOC: list[str] = [
    "bagaimana prakiraan cuaca besok pagi",
    "resep membuat kue lapis legit",
    "jadwal pertandingan liga sepak bola eropa",
    "harga tiket pesawat ke luar negeri",
]


# ---------------------------------------------------------------------------
# Penulisan
# ---------------------------------------------------------------------------


def write_pdf(path: Path, pages: dict[int, list[str]]) -> None:
    """Tulis satu PDF berteks (multi-halaman) dengan PyMuPDF."""
    doc = pymupdf.open()
    for page_no in sorted(pages):
        page = doc.new_page()
        y = 72.0
        for line in pages[page_no]:
            page.insert_text((72, y), line, fontname="helv", fontsize=12)
            y += 18.0
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    doc.close()


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> int:
    """Bangkitkan seluruh data sandbox Fase 5."""
    base = data_io.resolve_path(data_io.load_config()["paths"]["sandbox"]).parent
    pdf_dir = base / "pdf"
    eval_dir = base / "rag_eval"

    # Bersihkan PDF sandbox lama agar idempoten.
    if pdf_dir.exists():
        for old in pdf_dir.glob("*.pdf"):
            old.unlink()

    n_pages = 0
    for name, pages in PAGES.items():
        write_pdf(pdf_dir / name, pages)
        n_pages += len(pages)

    _write_csv(eval_dir / "qa_pairs.csv", ["question", "doc", "page"], [list(r) for r in QA_PAIRS])
    _write_csv(eval_dir / "in_doc.csv", ["question"], [[q] for q in IN_DOC])
    _write_csv(eval_dir / "out_doc.csv", ["question"], [[q] for q in OUT_DOC])

    print(f"pdf  : {pdf_dir} ({n_pages} halaman, {len(PAGES)} dokumen)")
    print(f"eval : {eval_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
