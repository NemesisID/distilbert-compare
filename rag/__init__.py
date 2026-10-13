"""Pipeline PDF/retrieval untuk chatbot Inkubator Bisnis UPN "Veteran" Jatim.

Modul:
- ``pdf_extract`` : ekstraksi teks per halaman dari PDF (PyMuPDF).
- ``chunking``    : pemotongan teks menjadi potongan (chunk) berulang.
- ``textnorm``    : normalisasi & tokenisasi (BM25 ``lowercase_space``).
- ``index``       : pembuatan/penyimpanan indeks (dense + BM25).
- ``retrieve``    : pencarian dense / BM25 / fusi RRF.
- ``eval_retrieval`` : evaluasi retrieval pada data rag_eval + pemilihan tau2.
"""
