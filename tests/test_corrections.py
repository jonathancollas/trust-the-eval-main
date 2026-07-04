"""Tests for the declared correction policy.

These lock the policy that used to live, unstated, inside a one-off generator:
each error_type maps to a declared action, drops are explicit (never silently
scored against a bad key), and every action carries a source.
"""
import pytest

from trust_the_eval.calibration.realworld import (AMBIGUITY_ERROR_TYPES,
                                                  DEFECT_ERROR_TYPES)
from trust_the_eval.corrections import (CORRECTION_IS_CANDIDATE, apply_policy,
                                        parse_correct_answer, policy_table)

CH = ["14N", "2H", "19F", "6Li"]


def test_policy_reproduces_the_real_rule():
    # wrong_groundtruth + a parseable correction -> re-key (change)
    d = apply_policy("Wrong groundtruth", "B", "A", CH)
    assert d.action == "change" and d.corrected_gold == ["A"] and d.scored and d.changed
    # wrong_groundtruth with no correction -> drop
    d = apply_policy("wrong_groundtruth", "B", "", CH)
    assert d.action == "drop" and d.corrected_gold is None and not d.scored
    # no_correct_answer -> drop
    d = apply_policy("no_correct_answer", "C", None, CH)
    assert d.action == "drop" and not d.scored
    # multiple_correct_answers -> union of original + parsed
    d = apply_policy("multiple correct answers", "A", "C", CH)
    assert d.action == "union" and d.corrected_gold == ["A", "C"] and d.scored and d.changed
    # multiple with nothing extra parsed -> unchanged set {original}
    d = apply_policy("multiple_correct_answers", "A", "", CH)
    assert d.action == "union" and d.corrected_gold == ["A"] and not d.changed
    # ok / clarity / expert -> keep, scored, unchanged
    for et in ("ok", "bad_question_clarity", "bad_options_clarity", "expert"):
        d = apply_policy(et, "D", None, CH)
        assert d.action == "keep" and d.corrected_gold == ["D"] and d.scored and not d.changed
    # unknown error type -> conservative keep
    d = apply_policy("totally_unknown", "A", None, CH)
    assert d.action == "keep" and d.scored


def test_drops_are_explicit_and_never_scored():
    for et in ("no_correct_answer",):
        d = apply_policy(et, "A", None, CH)
        assert d.scored is False and d.corrected_gold is None and d.changed is False
    # invariant: a scored decision always has a usable key; a dropped one never does
    for et in ("ok", "wrong_groundtruth", "no_correct_answer", "multiple_correct_answers",
               "bad_question_clarity", "expert", "unknown"):
        for ca in (None, "", "A", "2"):
            d = apply_policy(et, "A", ca, CH)
            assert (d.corrected_gold is not None) == d.scored


def test_every_defect_type_changes_unions_or_drops_never_silent_keep():
    for et in DEFECT_ERROR_TYPES:
        actions = {apply_policy(et, "A", ca, CH).action for ca in (None, "", "B")}
        assert actions <= {"change", "union", "drop"}, (et, actions)
        assert "keep" not in actions, et
    for et in AMBIGUITY_ERROR_TYPES:
        assert apply_policy(et, "A", None, CH).action == "keep", et


def test_table_has_a_source_for_every_row_and_covers_the_taxonomy():
    rows = policy_table()
    covered = {r["error_type"] for r in rows}
    for et in DEFECT_ERROR_TYPES | AMBIGUITY_ERROR_TYPES | {"ok"}:
        assert et in covered, et
    for r in rows:
        assert r["source"] and len(r["source"]) > 10
        assert r["action"] in ("keep", "change", "union", "drop")
        assert (r["action"] != "drop") == r["scored"]


def test_parse_correct_answer_variants():
    assert parse_correct_answer("0", CH) == {"A"}      # numeric index, base 0
    assert parse_correct_answer("2", CH) == {"C"}
    assert parse_correct_answer("A", CH) == {"A"}      # bare letter
    assert parse_correct_answer("2H", CH) == {"B"}     # exact option text
    assert parse_correct_answer("", CH) is None
    assert parse_correct_answer(None, CH) is None


def test_change_falls_back_to_drop_without_a_correction():
    d = apply_policy("wrong_groundtruth", "C", None, CH)
    assert d.action == "drop" and not d.scored


def test_correction_is_declared_a_candidate_not_truth():
    assert CORRECTION_IS_CANDIDATE and "candidate" in CORRECTION_IS_CANDIDATE.lower()
    assert "never asserts" in CORRECTION_IS_CANDIDATE.lower()


def test_decision_is_serialisable_with_provenance():
    d = apply_policy("Wrong groundtruth", "B", "A", CH).to_dict()
    for k in ("raw_error_type", "error_type", "action", "original_gold",
              "corrected_gold", "scored", "changed", "source", "note"):
        assert k in d
    assert d["raw_error_type"] == "Wrong groundtruth" and d["error_type"] == "wrong_groundtruth"
