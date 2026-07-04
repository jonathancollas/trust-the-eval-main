"""Tests for the MMLU-Redux corrections import adapter.

The adapter must *defer* the gold decision to the declared policy (so its output equals
``apply_policy``), build a canonical item id that the predictions half joins on, and
journal every transformation with provenance — nothing silent, nothing re-implemented.
"""
from trust_the_eval.corrections import apply_policy
from trust_the_eval.ingest import (canonical_item_id, gold_map,
                                    ingest_mmlu_redux, stamp_predictions)

ROWS = [
    {"question": "Q one", "choices": ["a", "b", "c", "d"], "answer": 1,
     "error_type": "wrong_groundtruth", "correct_answer": "A",
     "potential_reason": "key wrong", "subject": "college_chemistry"},
    {"question": "Q two", "choices": ["a", "b", "c", "d"], "answer": 0,
     "error_type": "no_correct_answer", "correct_answer": "", "subject": "college_chemistry"},
    {"question": "Q three", "choices": ["a", "b", "c", "d"], "answer": 0,
     "error_type": "multiple_correct_answers", "correct_answer": "C", "subject": "virology"},
    {"question": "Q four", "choices": ["a", "b", "c", "d"], "answer": 3,
     "error_type": "ok", "subject": "virology"},
]


def test_adapter_decisions_equal_the_declared_policy():
    corr, _intr, _j = ingest_mmlu_redux(ROWS)
    for raw, rec in zip(ROWS, corr):
        og = "ABCD"[int(raw["answer"])]
        dec = apply_policy(raw["error_type"], og, raw.get("correct_answer"), raw["choices"])
        assert rec.action == dec.action
        assert rec.corrected_gold == dec.corrected_gold
        assert rec.scored == dec.scored
        assert rec.changed == dec.changed


def test_canonical_id_is_stable_and_joins_the_two_halves():
    corr, _i, _j = ingest_mmlu_redux(ROWS)
    rec = corr[0]
    # a predictions adapter, using the same id function, lands on the same id
    pid = canonical_item_id("college_chemistry", "Q one", 1)
    assert pid == rec.item
    stamped, dropped = stamp_predictions(
        [{"item": pid, "subject": "college_chemistry", "preds": {"m": "A"}}], gold_map(corr))
    assert len(stamped) == 1 and stamped[0]["corrected_gold"] == ["A"] and not dropped


def test_journal_logs_every_transformation():
    _c, _i, journal = ingest_mmlu_redux(ROWS)
    j0 = journal[0]
    fields = {t["field"] for t in j0["transforms"]}
    assert {"answer", "error_type", "correct_answer"} <= fields
    answer_t = next(t for t in j0["transforms"] if t["field"] == "answer")
    assert answer_t["raw"] == 1 and answer_t["normalized"] == "B"     # index → letter logged
    assert j0["original_gold"] == ["B"] and j0["corrected_gold"] == ["A"]


def test_dropped_item_is_marked_and_excluded():
    corr, _i, _j = ingest_mmlu_redux(ROWS)
    drop = next(r for r in corr if r.error_type == "no_correct_answer")
    assert drop.scored is False and drop.corrected_gold is None
    gm = gold_map(corr)
    assert gm[drop.item]["scored"] is False
    _stamped, dropped = stamp_predictions(
        [{"item": drop.item, "preds": {"m": "A"}}], gm)
    assert drop.item in dropped


def test_provenance_is_preserved_on_every_record():
    corr, _i, _j = ingest_mmlu_redux(ROWS)
    p = corr[0].provenance
    assert p["source"].startswith("MMLU-Redux")
    assert p["raw"]["error_type"] == "wrong_groundtruth"
    assert p["raw"]["correct_answer"] == "A"
    assert "decision" in p and p["decision"]["action"] == "change"
    assert "candidate" in corr[0].candidate_note.lower()
    assert corr[0].potential_reason == "key wrong"


def test_intrinsic_row_shape():
    _c, intr, _j = ingest_mmlu_redux(ROWS)
    r = intr[0]
    assert set(r.keys()) == {"question", "choices", "answer", "error_type", "subject"}
    assert r["answer"] == 1 and r["subject"] == "college_chemistry"


def test_field_map_tolerance():
    rows = [{"q": "Q", "opts": ["a", "b", "c", "d"], "gold": 2,
             "err": "wrong_groundtruth", "fix": "A", "subj": "anatomy"}]
    fm = {"question": "q", "choices": "opts", "answer": "gold",
          "error_type": "err", "correct_answer": "fix", "subject": "subj"}
    corr, _i, _j = ingest_mmlu_redux(rows, field_map=fm)
    assert corr[0].subject == "anatomy"
    assert corr[0].original_gold == ["C"] and corr[0].corrected_gold == ["A"]
    assert corr[0].action == "change"


def test_subject_override_when_missing():
    rows = [{"question": "Q", "choices": ["a", "b", "c", "d"], "answer": 0, "error_type": "ok"}]
    corr, _i, _j = ingest_mmlu_redux(rows, subject="formal_logic")
    assert corr[0].subject == "formal_logic"


# --- predictions adapter (lm-eval --log_samples) ------------------------------ #

from trust_the_eval.ingest import ingest_lm_eval
from trust_the_eval.result_sensitivity import sensitivity
from trust_the_eval.transparency import _pog


def _mc(q, ans_idx, lps, subject="college_chemistry"):
    return {"doc": {"question": q, "choices": ["w", "x", "y", "z"], "answer": ans_idx,
                    "subject": subject},
            "target": ans_idx, "filtered_resps": [[v, False] for v in lps]}


def test_lm_eval_argmax_loglikelihood():
    rows, journal = ingest_lm_eval({"m1": [_mc("Q", 1, [-3.0, -0.5, -4.0, -5.0])]})
    assert rows[0]["preds"]["m1"] == "B"          # argmax index 1
    assert journal[0]["rule"].startswith("argmax")


def test_lm_eval_generative_parse():
    s = {"doc": {"question": "Q", "choices": ["w", "x", "y", "z"], "answer": 0},
         "filtered_resps": ["The answer is C."]}
    rows, _j = ingest_lm_eval({"m1": [s]})
    assert rows[0]["preds"]["m1"] == "C"


def test_lm_eval_merges_models_and_joins_corrections():
    Q = "Some question text"
    samples = {
        "claude": [_mc(Q, 1, [-3, -0.5, -4, -5])],   # B
        "gemini": [_mc(Q, 1, [-0.4, -2, -4, -5])],   # A
    }
    pred_rows, _j = ingest_lm_eval(samples)
    assert set(pred_rows[0]["preds"]) == {"claude", "gemini"}
    # the corrections half on the same question/index lands on the same id
    corr, _i, _jj = ingest_mmlu_redux(
        [{"question": Q, "choices": ["w", "x", "y", "z"], "answer": 1,
          "error_type": "wrong_groundtruth", "correct_answer": "A", "subject": "college_chemistry"}])
    assert pred_rows[0]["item"] == corr[0].item
    stamped, dropped = stamp_predictions(
        [{"item": r["item"], "subject": r["subject"], "preds": r["preds"]} for r in pred_rows],
        gold_map(corr))
    assert stamped[0]["original_gold"] == ["B"] and stamped[0]["corrected_gold"] == ["A"]
    assert not dropped


def test_end_to_end_import_feeds_sensitivity():
    # two items, three models, one key correction -> the joined rows must score.
    # gold index is fixed per question (the benchmark's gold); only loglikelihoods vary.
    QA, QB = "Question A", "Question B"
    samples = {
        "m1": [_mc(QA, 1, [-3, -0.5, -4, -5]), _mc(QB, 0, [-0.4, -2, -3, -4])],  # A:B  B:A
        "m2": [_mc(QA, 1, [-0.4, -2, -4, -5]), _mc(QB, 0, [-0.5, -2, -3, -4])],  # A:A  B:A
        "m3": [_mc(QA, 1, [-0.4, -2, -4, -5]), _mc(QB, 0, [-3, -3, -0.4, -4])],  # A:A  B:C
    }
    pred_rows, _pj = ingest_lm_eval(samples)
    assert len(pred_rows) == 2 and all(len(r["preds"]) == 3 for r in pred_rows)
    corr, intr, _cj = ingest_mmlu_redux([
        {"question": QA, "choices": ["w", "x", "y", "z"], "answer": 1,
         "error_type": "wrong_groundtruth", "correct_answer": "A", "subject": "college_chemistry"},
        {"question": QB, "choices": ["w", "x", "y", "z"], "answer": 0,
         "error_type": "ok", "subject": "college_chemistry"},
    ])
    scored, _d = stamp_predictions(
        [{"item": r["item"], "subject": r["subject"], "preds": r["preds"]} for r in pred_rows],
        gold_map(corr))
    p, o, c = _pog(scored)[:3]
    res = sensitivity(p, o, c, iters=200, seed=0)
    assert res["n_items_scored"] == 2 and res["n_models"] == 3
    assert res["n_changed_items"] == 1          # only QA's key changed (B->A)


# --- HELM adapter (scenario_state.json) and other harness-family troves --------- #

from trust_the_eval.ingest import ingest_helm

_Q = "Some HELM question"


def _helm(letter, correct_idx=1):
    refs = [{"output": {"text": t}, "tags": (["correct"] if i == correct_idx else [])}
            for i, t in enumerate(["14N", "2H", "19F", "6Li"])]
    return {"request_states": [{"instance": {"input": {"text": _Q}, "references": refs},
                                "result": {"completions": [{"text": letter}]}}]}


def test_helm_reads_correct_tag_and_completion():
    rows, journal = ingest_helm({"m1": _helm("B"), "m2": _helm(" A")}, subject="college_chemistry")
    assert rows[0]["original_gold"] == ["B"]             # correct tag on index 1
    assert rows[0]["preds"] == {"m1": "B", "m2": "A"}
    assert journal[0]["rule"].startswith("letter")


def test_helm_text_match_completion():
    # completion is the answer text, not a letter -> matched to the reference's option
    rows, _j = ingest_helm({"m": _helm("2H")}, subject="college_chemistry")
    assert rows[0]["preds"]["m"] == "B"


def test_helm_skips_when_no_correct_reference():
    bad = {"request_states": [{"instance": {"input": {"text": _Q},
            "references": [{"output": {"text": "x"}, "tags": []}]},
            "result": {"completions": [{"text": "A"}]}}]}
    rows, journal = ingest_helm({"m": bad}, subject="demo")
    assert rows == []                                    # nothing scorable
    assert journal[0]["item"] is None and "correct" in journal[0]["rule"]


def test_helm_joins_corrections_and_feeds_sensitivity():
    samples = {"m1": _helm("B"), "m2": _helm("A"), "m3": _helm("A")}
    pred_rows, _j = ingest_helm(samples, subject="college_chemistry")
    corr, intr, _jj = ingest_mmlu_redux(
        [{"question": _Q, "choices": ["14N", "2H", "19F", "6Li"], "answer": 1,
          "error_type": "wrong_groundtruth", "correct_answer": "A", "subject": "college_chemistry"}])
    assert pred_rows[0]["item"] == corr[0].item          # HELM id joins the corrections id
    scored, _d = stamp_predictions(
        [{"item": r["item"], "subject": r["subject"], "preds": r["preds"]} for r in pred_rows],
        gold_map(corr))
    assert scored[0]["original_gold"] == ["B"] and scored[0]["corrected_gold"] == ["A"]
    p, o, c = _pog(scored)[:3]
    res = sensitivity(p, o, c, iters=200, seed=0)
    assert res["n_items_scored"] == 1 and res["n_models"] == 3 and res["n_changed_items"] == 1


def test_lm_eval_covers_open_llm_leaderboard_details():
    # Open LLM Leaderboard v1 'details' rows use example/gold/predictions field names
    details = {"m1": [{"example": "Q", "choices": ["w", "x", "y", "z"], "gold": 2,
                       "predictions": [[-3, False], [-4, False], [-0.4, False], [-5, False]]}]}
    rows, _j = ingest_lm_eval(details)
    assert rows[0]["preds"]["m1"] == "C"                 # argmax index 2


# --- Inspect adapter (.eval / EvalLog JSON) ------------------------------------ #

from trust_the_eval.ingest import ingest_inspect

_IQ = "Some Inspect question"
_ICH = ["14N", "2H", "19F", "6Li"]


def _isample(pred, target="B", via_score=True):
    s = {"input": "prompt text", "choices": _ICH, "target": target,
         "metadata": {"question": _IQ, "subject": "college_chemistry"}}
    if via_score:
        s["scores"] = {"choice": {"value": "C", "answer": pred}}
    else:
        s["output"] = {"completion": "The answer is %s." % pred}
    return s


def test_inspect_reads_target_and_score_answer():
    rows, journal = ingest_inspect({"m1": {"samples": [_isample("A")]}}, subject="college_chemistry")
    assert rows[0]["original_gold"] == ["B"]             # target letter
    assert rows[0]["preds"]["m1"] == "A"                 # from the choice scorer's answer
    assert "score answer" in journal[0]["rule"]


def test_inspect_completion_fallback():
    rows, _j = ingest_inspect({"m1": {"samples": [_isample("C", via_score=False)]}}, subject="x")
    assert rows[0]["preds"]["m1"] == "C"


def test_inspect_target_as_index():
    s = {"choices": _ICH, "target": 1, "metadata": {"question": _IQ, "subject": "x"},
         "output": {"completion": "D"}}
    rows, _j = ingest_inspect({"m": {"samples": [s]}}, subject="x")
    assert rows[0]["original_gold"] == ["B"] and rows[0]["preds"]["m"] == "D"


def test_inspect_question_from_messages():
    s = {"input": [{"role": "system", "content": "sys"},
                   {"role": "user", "content": "the real question"}],
         "choices": _ICH, "target": "A"}
    rows, _j = ingest_inspect({"m": {"samples": [s]}}, subject="x")
    assert "the real question" in rows[0]["item"]        # question recovered from messages


def test_inspect_joins_corrections_and_feeds_sensitivity():
    logs = {"m1": {"samples": [_isample("B")]},
            "m2": {"samples": [_isample("A")]},
            "m3": {"samples": [_isample("A", via_score=False)]}}
    pred_rows, _j = ingest_inspect(logs, subject="college_chemistry")
    corr, intr, _jj = ingest_mmlu_redux(
        [{"question": _IQ, "choices": _ICH, "answer": 1,
          "error_type": "wrong_groundtruth", "correct_answer": "A", "subject": "college_chemistry"}])
    assert pred_rows[0]["item"] == corr[0].item          # Inspect id joins the corrections id
    scored, _d = stamp_predictions(
        [{"item": r["item"], "subject": r["subject"], "preds": r["preds"]} for r in pred_rows],
        gold_map(corr))
    assert scored[0]["original_gold"] == ["B"] and scored[0]["corrected_gold"] == ["A"]
    p, o, c = _pog(scored)[:3]
    res = sensitivity(p, o, c, iters=200, seed=0)
    assert res["n_items_scored"] == 1 and res["n_models"] == 3 and res["n_changed_items"] == 1


# --- Platinum corrections + free-form predictions (a second, non-MCQ corpus) ---- #

from trust_the_eval.ingest import (ingest_lm_eval_freeform, ingest_platinum,
                                   normalize_answer)

PLAT = [
    {"question": "Q one", "cleaning_status": "consensus",
     "platinum_target": ["3"], "original_target": ["3"], "subject": "gsm8k"},
    {"question": "Q two", "cleaning_status": "revised",
     "platinum_target": ["42"], "original_target": ["41"], "subject": "gsm8k"},
    {"question": "Q three", "cleaning_status": "rejected",
     "platinum_target": ["7"], "original_target": ["7"], "subject": "gsm8k"},
    {"question": "Q four", "cleaning_status": "verified",
     "platinum_target": ["9"], "original_target": ["9"], "subject": "gsm8k"},
]


def test_platinum_status_maps_to_keep_change_drop():
    corr, _i, _j = ingest_platinum(PLAT, subject="gsm8k")
    by_status = {r.error_type: r for r in corr}
    assert by_status["consensus"].action == "keep" and by_status["consensus"].scored
    assert by_status["verified"].action == "keep"
    assert by_status["revised"].action == "change"
    assert by_status["revised"].corrected_gold == ["42"] and by_status["revised"].changed
    assert by_status["rejected"].action == "drop" and by_status["rejected"].scored is False
    # golds are answer strings, not letters
    assert by_status["revised"].original_gold == ["41"]


def test_freeform_answer_normalization():
    assert normalize_answer("Answer: 42") == "42"
    assert normalize_answer("  '42'. ") == "42"
    assert normalize_answer(["42"][0]) == "42"


def test_canonical_id_freeform_has_no_index_suffix():
    fid = canonical_item_id("gsm8k", "What is 17 plus 25?", None)
    assert fid.count("::") == 1 and fid.endswith("plus 25?")


def test_platinum_and_freeform_predictions_join_and_feed_atlas():
    corr, intr, _j = ingest_platinum(PLAT, subject="gsm8k")

    def s(q, ans):
        return {"doc": {"question": q, "subject": "gsm8k"}, "filtered_resps": ["Answer: %s" % ans]}
    Q1, Q2 = "Q one", "Q two"
    samples = {"m1": [s(Q1, "3"), s(Q2, "42")],   # both right after correction
               "m2": [s(Q1, "3"), s(Q2, "41")],   # Q2 matches the OLD key only
               "m3": [s(Q1, "4"), s(Q2, "42")]}   # Q1 wrong
    pred_rows, _pj = ingest_lm_eval_freeform(samples, subject="gsm8k")
    # the free-form prediction id joins the Platinum correction id (same question)
    corr_q2 = next(r for r in corr if r.error_type == "revised")
    assert any(r["item"] == corr_q2.item for r in pred_rows)
    scored, _d = stamp_predictions(
        [{"item": r["item"], "subject": r["subject"], "preds": r["preds"]} for r in pred_rows],
        gold_map(corr))
    # the revised item now scores against the corrected string gold
    q2 = next(r for r in scored if r["item"] == corr_q2.item)
    assert q2["original_gold"] == ["41"] and q2["corrected_gold"] == ["42"]
    from trust_the_eval.atlas import cell_metrics
    m = cell_metrics("PLATINUM::gsm8k", scored, [c.intrinsic() for c in corr], iters=200, seed=0)
    assert m["n_changed"] == 1 and m["kendall_tau"] is not None

