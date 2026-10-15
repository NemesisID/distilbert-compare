"""Uji routing (``src/router.py``) tanpa memuat model/jaringan."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from src import router as R
from src.answers import (
    load_answer_variants,
    load_answers,
    normalize_variants,
    pick_answer,
)


# ---------------------------------------------------------------------------
# decide_route (murni)
# ---------------------------------------------------------------------------


def _decide(**over):
    base = dict(
        intent="sapaan",
        confidence=0.9,
        oos_intent="out_of_scope",
        doc_intent="tanya_dokumen",
        tau1=0.5,
        retrieval_score=None,
        tau2=0.3,
    )
    base.update(over)
    return R.decide_route(**base)


def test_decide_oos_goes_fallback():
    assert _decide(intent="out_of_scope") == (R.ROUTE_FALLBACK, "oos")


def test_decide_low_confidence_goes_fallback():
    assert _decide(confidence=0.2) == (R.ROUTE_FALLBACK, "low_confidence")


def test_decide_generic_intent_goes_faq():
    assert _decide(intent="sapaan") == (R.ROUTE_FAQ, "faq_answer")


def test_decide_doc_with_score_above_tau2_goes_dokumen():
    out = _decide(intent="tanya_dokumen", retrieval_score=0.6)
    assert out == (R.ROUTE_DOKUMEN, "retrieval_hit")


def test_decide_doc_with_score_below_tau2_goes_fallback():
    out = _decide(intent="tanya_dokumen", retrieval_score=0.1)
    assert out == (R.ROUTE_FALLBACK, "doc_not_found")


def test_decide_doc_without_retrieval_goes_fallback():
    out = _decide(intent="tanya_dokumen", retrieval_score=None)
    assert out == (R.ROUTE_FALLBACK, "no_retrieval")


def test_decide_doc_without_tau2_accepts_score():
    out = _decide(intent="tanya_dokumen", retrieval_score=0.01, tau2=None)
    assert out == (R.ROUTE_DOKUMEN, "retrieval_hit")


# ---------------------------------------------------------------------------
# load_answers / load_answer_variants / normalize_variants / pick_answer
# ---------------------------------------------------------------------------


def test_load_answers_reads_yaml(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text(
        "- intent: sapaan\n  jawaban: Halo\n- intent: terima_kasih\n  jawaban: Sama-sama\n",
        encoding="utf-8",
    )
    assert load_answers(p) == {"sapaan": "Halo", "terima_kasih": "Sama-sama"}


def test_load_answers_missing_file_returns_empty(tmp_path):
    assert load_answers(tmp_path / "tidak_ada.yaml") == {}


def test_load_answer_variants_supports_string_and_list(tmp_path):
    p = tmp_path / "v.yaml"
    p.write_text(
        "- intent: sapaan\n  jawaban: Halo\n"
        "- intent: terima_kasih\n  jawaban:\n    - Sama-sama\n    - Dengan senang hati\n",
        encoding="utf-8",
    )
    assert load_answer_variants(p) == {
        "sapaan": ["Halo"],
        "terima_kasih": ["Sama-sama", "Dengan senang hati"],
    }
    # Jawaban utama = variasi pertama (tetap kompatibel).
    assert load_answers(p) == {"sapaan": "Halo", "terima_kasih": "Sama-sama"}


def test_load_answer_variants_skips_entry_without_answer(tmp_path):
    p = tmp_path / "v.yaml"
    p.write_text(
        "- intent: sapaan\n  deskripsi: belum diisi\n"
        "- intent: terima_kasih\n  jawaban: OK\n",
        encoding="utf-8",
    )
    assert load_answer_variants(p) == {"terima_kasih": ["OK"]}


def test_normalize_variants_rules():
    assert normalize_variants(None) == []
    assert normalize_variants("") == []
    assert normalize_variants("   ") == []
    assert normalize_variants(" Halo ") == ["Halo"]
    assert normalize_variants(["A", " ", "B", None]) == ["A", "B"]
    assert normalize_variants(5) == ["5"]


def test_pick_answer_is_deterministic_without_rng():
    assert pick_answer(["A", "B", "C"]) == "A"  # selalu variasi pertama
    assert pick_answer([]) == ""
    assert pick_answer(["  ", ""]) == ""


def test_pick_answer_single_variant_ignores_rng():
    assert pick_answer(["A"], random.Random(1)) == "A"


def test_pick_answer_varies_with_seeded_rng():
    rng = random.Random(42)
    picks = {pick_answer(["A", "B", "C"], rng) for _ in range(60)}
    assert picks == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# Router.route dengan metode palsu (tanpa model)
# ---------------------------------------------------------------------------


def _fake_router(answers=None, variants=None, variant_seed=42):
    r = R.Router.__new__(R.Router)
    r.routing = {
        "oos_intent": "out_of_scope",
        "doc_intent": "tanya_dokumen",
        "fallback_answer": "FALLBACK",
        "not_found_answer": "NOT_FOUND",
    }
    r.answers = {"sapaan": "Halo!"} if answers is None else answers
    r.answer_variants = (
        {k: [v] for k, v in r.answers.items()} if variants is None else variants
    )
    r.answer_variants_enabled = True
    r._rng = random.Random(variant_seed)
    r.tau1 = 0.5
    r.tau2 = 0.3
    return r


def test_route_faq_uses_canned_answer():
    r = _fake_router()
    r.classify = lambda text: ("sapaan", 0.9)
    r.retrieve = lambda text: (None, [])
    d = r.route("halo")
    assert d.route == R.ROUTE_FAQ
    assert d.answer == "Halo!"


def test_route_faq_picks_variant_from_list():
    r = _fake_router(variants={"sapaan": ["Halo!", "Selamat datang!"]}, variant_seed=42)
    r.classify = lambda text: ("sapaan", 0.9)
    r.retrieve = lambda text: (None, [])
    picks = {r.route("halo").answer for _ in range(40)}
    assert picks == {"Halo!", "Selamat datang!"}


def test_route_faq_uses_primary_when_variants_disabled():
    r = _fake_router(variants={"sapaan": ["Halo!", "Selamat datang!"]})
    r.answer_variants_enabled = False
    r.classify = lambda text: ("sapaan", 0.9)
    r.retrieve = lambda text: (None, [])
    assert {r.route("halo").answer for _ in range(10)} == {"Halo!"}


def test_route_faq_falls_back_when_intent_has_no_answer():
    r = _fake_router(answers={}, variants={})
    r.classify = lambda text: ("sapaan", 0.9)
    r.retrieve = lambda text: (None, [])
    assert r.route("halo").answer == "FALLBACK"


def test_route_document_builds_citation():
    r = _fake_router()
    r.classify = lambda text: ("tanya_dokumen", 0.9)
    r.retrieve = lambda text: (0.8, [{"doc": "a.pdf", "page": 2, "score": 0.8,
                                      "dense_score": 0.8, "bm25_score": 1.0, "text": "isi dokumen"}])
    d = r.route("apa isi dokumen")
    assert d.route == R.ROUTE_DOKUMEN
    assert "a.pdf" in d.answer and "hal. 2" in d.answer
    assert d.sources and d.sources[0]["page"] == 2


def test_route_document_not_found_uses_not_found_answer():
    r = _fake_router()
    r.classify = lambda text: ("tanya_dokumen", 0.9)
    r.retrieve = lambda text: (0.05, [{"doc": "a.pdf", "page": 1, "score": 0.05,
                                       "dense_score": 0.05, "bm25_score": 0.0, "text": "x"}])
    d = r.route("apa isi dokumen")
    assert d.route == R.ROUTE_FALLBACK
    assert d.reason == "doc_not_found"
    assert d.answer == "NOT_FOUND"


def test_route_oos_uses_fallback_answer():
    r = _fake_router()
    r.classify = lambda text: ("out_of_scope", 0.99)
    r.retrieve = lambda text: (None, [])
    d = r.route("cuaca besok")
    assert d.route == R.ROUTE_FALLBACK
    assert d.answer == "FALLBACK"


# ---------------------------------------------------------------------------
# evaluate (dengan router palsu)
# ---------------------------------------------------------------------------


class _StubRouter:
    def __init__(self, mapping):
        self.mapping = mapping

    def route(self, text):
        return R.RouteDecision(
            route=self.mapping[text], intent="x", confidence=1.0, answer="", reason="stub"
        )


def test_evaluate_accuracy_and_confusion():
    rows = [
        {"question": "q1", "expected_route": "faq"},
        {"question": "q2", "expected_route": "dokumen"},
        {"question": "q3", "expected_route": "fallback"},
    ]
    stub = _StubRouter({"q1": "faq", "q2": "fallback", "q3": "fallback"})
    metrics = R.evaluate(stub, rows, ["faq", "dokumen", "fallback"])
    assert metrics["n"] == 3
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["confusion"]["dokumen"]["fallback"] == 1
    assert metrics["per_route"]["faq"]["f1"] == pytest.approx(1.0)
