"""Tests for trust_the_eval.corpus — the versioned annotation corpus (P1)."""
import os

from trust_the_eval.calibration.realworld import DEFECT_ERROR_TYPES
from trust_the_eval.corpus import AnnotationCorpus

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "mmlu_redux_sample.jsonl")


def test_synthetic_seed_measures():
    c = AnnotationCorpus.synthetic_seed()
    assert c.n == 48 and c.has_annotations
    le = c.label_error()
    assert 0 < le["rate"] < 1 and le["n"] == 48
    assert c.ambiguity()["rate"] > 0
    assert c.worst_subject()["subject"] == "virology"
    art = c.to_artifact()
    assert art.n == 48 and art.items[0].meta.get("category")


def test_content_hash_is_deterministic():
    a = AnnotationCorpus.synthetic_seed()
    b = AnnotationCorpus.synthetic_seed()
    assert a.content_hash() == b.content_hash()
    assert a.content_hash().startswith("sha256:")


def test_from_mmlu_redux_fixture_loads_and_measures():
    c = AnnotationCorpus.from_mmlu_redux(path=FIX)
    assert c.benchmark == "MMLU" and c.n == 10 and c.has_annotations
    exp = sum(1 for r in c.rows if r["error_type"] in DEFECT_ERROR_TYPES) / len(c.rows)
    assert abs(c.label_error()["rate"] - exp) < 1e-9
