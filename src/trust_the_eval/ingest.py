"""Import adapter — MMLU-Redux corrections (the head of the lineage thread).

This is steps 1→2 of a lineage thread, as real code: it reads a raw MMLU-Redux row
(``question``, ``choices``, ``answer``, ``error_type``, ``correct_answer``,
``potential_reason``) and turns it into a canonical correction record — **journaling
every transformation** and deferring the gold decision to the declared ``apply_policy``
(no rule is re-implemented here). Provenance is preserved on every record: the source,
the raw fields, the ordered transforms, and the policy decision.

It produces the *corrections half* (annotation + gold decision per item). A predictions
adapter produces the other half (model answers); the two join on the canonical item id
built here — one definition, so the halves line up exactly.

Field names match the edinburgh-dawg/mmlu-redux-2.0 schema and are overridable via
``field_map``. Network access to Hugging Face is not assumed; feed rows from a local
export (the CLI reads JSON / JSONL).

Zero-dependency; Python >= 3.8.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .calibration.realworld import canonical_error_type
from .corrections import CORRECTION_IS_CANDIDATE, _ACTION_SOURCE, apply_policy

LETTERS = "ABCD"
SOURCE_DEFAULT = "MMLU-Redux 2.0 (edinburgh-dawg; Gema et al. 2024)"
DEFAULT_FIELD_MAP = {
    "question": "question", "choices": "choices", "answer": "answer",
    "error_type": "error_type", "correct_answer": "correct_answer",
    "reason": "potential_reason", "subject": "subject",
}


def normalize_question(s: Any) -> str:
    s = (str(s) if s is not None else "").lower()
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'").replace("\u2018", "'")
    return re.sub(r"\s+", " ", s).strip()


def canonical_item_id(subject: Any, question: Any, answer_idx: Any, n: int = 80) -> str:
    """The one definition of an item id, shared by every adapter so the two halves join.

    MCQ: ``subject::<normalised-question-prefix>::<original-answer-index>``.
    Free-form (``answer_idx=None``): ``subject::<normalised-question-prefix>`` — there is
    no option index, so the question anchors the join.
    """
    base = "%s::%s" % (subject, normalize_question(question)[:n])
    if answer_idx is None:
        return base
    try:
        ai = int(answer_idx)
    except (TypeError, ValueError):
        ai = -1
    return "%s::%d" % (base, ai)


@dataclass
class Transform:
    field: str
    raw: Any
    normalized: Any
    rule: str

    def to_dict(self) -> Dict[str, Any]:
        return {"field": self.field, "raw": self.raw, "normalized": self.normalized, "rule": self.rule}


@dataclass
class CorrectionRecord:
    item: str
    subject: str
    question: str
    choices: List[str]
    answer: int                       # original MMLU index (0-3)
    error_type: Optional[str]         # raw
    potential_reason: Optional[str]
    original_gold: List[str]
    corrected_gold: Optional[List[str]]  # None when dropped
    scored: bool
    action: str
    changed: bool
    source: str                       # citation for the action taken
    candidate_note: str
    provenance: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item": self.item, "subject": self.subject, "question": self.question,
            "choices": self.choices, "answer": self.answer, "error_type": self.error_type,
            "potential_reason": self.potential_reason, "original_gold": self.original_gold,
            "corrected_gold": self.corrected_gold, "scored": self.scored,
            "action": self.action, "changed": self.changed, "source": self.source,
            "candidate_note": self.candidate_note, "provenance": self.provenance,
        }

    def intrinsic(self) -> Dict[str, Any]:
        """The intrinsic-annotation row the rest of the pipeline consumes."""
        return {"question": self.question, "choices": self.choices, "answer": self.answer,
                "error_type": self.error_type, "subject": self.subject}


def ingest_mmlu_redux(rows: Iterable[Dict[str, Any]], *, subject: Optional[str] = None,
                      source: str = SOURCE_DEFAULT, base: int = 0,
                      field_map: Optional[Dict[str, str]] = None, id_len: int = 80
                      ) -> Tuple[List[CorrectionRecord], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Turn raw MMLU-Redux rows into canonical correction records + intrinsic rows + a journal."""
    fm = {**DEFAULT_FIELD_MAP, **(field_map or {})}
    corrections: List[CorrectionRecord] = []
    intrinsic: List[Dict[str, Any]] = []
    journal: List[Dict[str, Any]] = []

    for raw in rows:
        q = raw.get(fm["question"])
        choices = list(raw.get(fm["choices"]) or [])
        ai = raw.get(fm["answer"])
        et = raw.get(fm["error_type"])
        ca = raw.get(fm["correct_answer"])
        reason = raw.get(fm["reason"])
        subj = raw.get(fm["subject"]) or subject or "unknown"

        try:
            ai_int = int(ai)
        except (TypeError, ValueError):
            ai_int = -1
        og_letter = LETTERS[ai_int] if 0 <= ai_int < 4 else None

        transforms = [
            Transform("answer", ai, og_letter, "original index → letter (0-3 → A-D)"),
            Transform("error_type", et, canonical_error_type(et), "canonicalise (lower / underscores)"),
        ]
        if ca not in (None, ""):
            transforms.append(Transform("correct_answer", ca, None,
                                        "passed to the correction policy parser"))

        # the gold decision is the declared policy's job — not re-implemented here
        dec = apply_policy(et, og_letter, ca, choices, base=base)
        item = canonical_item_id(subj, q, ai_int, n=id_len)

        rec = CorrectionRecord(
            item=item, subject=subj, question=q, choices=choices, answer=ai_int,
            error_type=et, potential_reason=reason,
            original_gold=dec.original_gold, corrected_gold=dec.corrected_gold,
            scored=dec.scored, action=dec.action, changed=dec.changed, source=dec.source,
            candidate_note=CORRECTION_IS_CANDIDATE,
            provenance={"source": source,
                        "raw": {"answer": ai, "error_type": et, "correct_answer": ca},
                        "transforms": [t.to_dict() for t in transforms],
                        "decision": dec.to_dict()})
        corrections.append(rec)
        intrinsic.append(rec.intrinsic())
        journal.append({"item": item, "subject": subj, "action": dec.action,
                        "scored": dec.scored, "original_gold": dec.original_gold,
                        "corrected_gold": dec.corrected_gold,
                        "transforms": [t.to_dict() for t in transforms]})
    return corrections, intrinsic, journal


def gold_map(corrections: Iterable[CorrectionRecord]) -> Dict[str, Dict[str, Any]]:
    """item -> {original_gold, corrected_gold, scored}; what a predictions adapter consumes."""
    return {r.item: {"original_gold": r.original_gold, "corrected_gold": r.corrected_gold,
                     "scored": r.scored} for r in corrections}


def stamp_predictions(pred_rows: Iterable[Dict[str, Any]], gmap: Dict[str, Dict[str, Any]]
                      ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Join raw prediction rows (item, preds) to the gold map, dropping non-scorable items.

    Demonstrates how the two halves connect: predictions carry the gold the corrections
    adapter decided, and items the policy dropped never enter the scored set.
    """
    out: List[Dict[str, Any]] = []
    dropped: List[str] = []
    for r in pred_rows:
        g = gmap.get(r["item"])
        if g is None:
            continue
        if not g["scored"]:
            dropped.append(r["item"])
            continue
        out.append({"item": r["item"], "subject": r.get("subject", ""),
                    "original_gold": g["original_gold"],
                    "corrected_gold": g["corrected_gold"] or g["original_gold"],
                    "preds": r.get("preds", {})})
    return out, dropped


def format_journal(journal: List[Dict[str, Any]], limit: Optional[int] = None,
                   source: str = "corrections") -> str:
    """Human-readable transform log — every change, per item."""
    out = ["import journal · %s · %d items" % (source, len(journal)), ""]
    rows = journal if limit is None else journal[:limit]
    for j in rows:
        out.append("%s  [%s%s]" % (j["item"], j["action"],
                                   "" if j["scored"] else ", dropped"))
        for t in j["transforms"]:
            out.append("    %-13s %r → %r   (%s)" % (t["field"], t["raw"], t["normalized"], t["rule"]))
        out.append("    gold          %r → %r" % (j["original_gold"], j["corrected_gold"]))
        out.append("")
    if limit is not None and len(journal) > limit:
        out.append("… %d more" % (len(journal) - limit))
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Free-form answers: normalisation shared by corrections and predictions       #
# --------------------------------------------------------------------------- #

def normalize_answer(s: Any) -> str:
    """Normalise a free-form answer so a gold and a prediction compare on equal terms.

    Strips an ``Answer:`` prefix (the Platinum prompt format), surrounding quotes and
    trailing punctuation, collapses whitespace, and lowercases. The SAME function must
    normalise both sides or nothing joins.
    """
    if s is None:
        return ""
    t = str(s).strip()
    low = t.lower()
    if "answer:" in low:
        t = t[low.rindex("answer:") + len("answer:"):]
    t = t.strip().strip("\"'` .").strip()
    return re.sub(r"\s+", " ", t).lower()


PLATINUM_STATUS_ACTION = {
    "consensus": "keep", "verified": "keep", "revised": "change", "rejected": "drop",
}
PLATINUM_SOURCE = "Platinum Benchmarks (MadryLab; Vendrow et al. 2025)"


def _plat_targets(v: Any) -> List[str]:
    if isinstance(v, (list, tuple)):
        return [normalize_answer(x) for x in v if x is not None and str(x) != ""]
    if v in (None, ""):
        return []
    return [normalize_answer(v)]


def ingest_platinum(rows: Iterable[Dict[str, Any]], *, subject: Optional[str] = None,
                    source: str = PLATINUM_SOURCE, field_map: Optional[Dict[str, str]] = None,
                    id_len: int = 80) -> Tuple[List[CorrectionRecord], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Turn Platinum revised rows into canonical (free-form) correction records.

    Platinum's ``cleaning_status`` is itself a declared policy — it maps one-to-one onto
    ours: ``consensus``/``verified`` → keep, ``revised`` → change (``platinum_target`` ≠
    ``original_target``), ``rejected`` → drop (removed for ambiguity). Golds are answer
    strings, so items join a free-form predictions run on the same questions.
    """
    fm = {"question": "question", "status": "cleaning_status",
          "platinum_target": "platinum_target", "original_target": "original_target",
          "subject": "subject", **(field_map or {})}
    corrections: List[CorrectionRecord] = []
    intrinsic: List[Dict[str, Any]] = []
    journal: List[Dict[str, Any]] = []

    for raw in rows:
        q = raw.get(fm["question"])
        status = str(raw.get(fm["status"]) or "").strip().lower()
        subj = raw.get(fm["subject"]) or subject or "unknown"
        plat = _plat_targets(raw.get(fm["platinum_target"]))
        orig = _plat_targets(raw.get(fm["original_target"])) or plat
        action = PLATINUM_STATUS_ACTION.get(status, "keep")

        if action == "drop":
            corrected, scored, changed = None, False, False
        elif action == "change":
            corrected, scored = sorted(set(plat)), True
            changed = set(orig) != set(plat)
        else:  # keep (consensus / verified / unknown)
            action, corrected, scored, changed = "keep", sorted(set(orig)), True, False

        item = canonical_item_id(subj, q, None, n=id_len)
        og_list = sorted(set(orig))
        transforms = [
            Transform("cleaning_status", raw.get(fm["status"]), action,
                      "Platinum revision status → policy action"),
            Transform("target", raw.get(fm["original_target"]), raw.get(fm["platinum_target"]),
                      "original_target → platinum_target"),
        ]
        rec = CorrectionRecord(
            item=item, subject=subj, question=q, choices=[], answer=-1,
            error_type=status, potential_reason=None,
            original_gold=og_list, corrected_gold=corrected, scored=scored, action=action,
            changed=changed, source=_ACTION_SOURCE.get(action, "") + " (Platinum revision)",
            candidate_note=CORRECTION_IS_CANDIDATE,
            provenance={"source": source,
                        "raw": {"cleaning_status": status,
                                "original_target": raw.get(fm["original_target"]),
                                "platinum_target": raw.get(fm["platinum_target"])},
                        "transforms": [t.to_dict() for t in transforms],
                        "decision": {"action": action, "scored": scored, "changed": changed}})
        corrections.append(rec)
        intrinsic.append({"question": q, "choices": [], "answer": -1,
                          "error_type": status, "subject": subj})
        journal.append({"item": item, "subject": subj, "action": action, "scored": scored,
                        "original_gold": og_list, "corrected_gold": corrected,
                        "transforms": [t.to_dict() for t in transforms]})
    return corrections, intrinsic, journal


# --------------------------------------------------------------------------- #
# Predictions adapter — lm-evaluation-harness --log_samples (the other half)   #
# --------------------------------------------------------------------------- #

PRED_FIELD_MAP = {
    "question": ["question", "query", "input", "example"],
    "choices": ["choices", "options", "endings"],
    "answer": ["answer", "gold", "label"],
    "subject": ["subject", "subtask", "task", "category"],
    "resps": ["filtered_resps", "resps", "predictions"],
}


def _first(d: Dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _as_logprob(x: Any) -> Optional[float]:
    if isinstance(x, (list, tuple)) and x:
        x = x[0]
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _letter_from_text(t: str) -> Optional[str]:
    """A standalone A-D letter (not one buried inside a word like 'Answer'); last wins."""
    ms = re.findall(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", t.upper())
    return ms[-1] if ms else None


def predicted_letter(sample: Dict[str, Any], n_choices: int,
                     resps_keys: Iterable[str]) -> Tuple[Optional[str], str]:
    """The model's chosen letter for one logged sample, plus how it was decoded.

    Handles the two common lm-eval shapes: multiple-choice loglikelihoods (argmax over
    the per-option scores) and a generated string (a standalone A-D letter).
    """
    resps = _first(sample, resps_keys)
    if isinstance(resps, list) and resps:
        lp = [_as_logprob(r) for r in resps]
        if n_choices and len(lp) == n_choices and all(v is not None for v in lp):
            best = max(range(n_choices), key=lambda i: lp[i])
            return LETTERS[best], "argmax loglikelihood over %d choices" % n_choices
        for r in resps:
            if isinstance(r, str):
                lt = _letter_from_text(r)
                if lt:
                    return lt, "parsed letter from generation"
    elif isinstance(resps, str):
        lt = _letter_from_text(resps)
        if lt:
            return lt, "parsed letter from generation"
    return None, "unparseable response"


def ingest_lm_eval(samples_by_model: Dict[str, Iterable[Dict[str, Any]]], *,
                   subject: Optional[str] = None, field_map: Optional[Dict[str, List[str]]] = None,
                   id_len: int = 80) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Merge per-model lm-eval ``--log_samples`` files into canonical prediction rows.

    Returns ``(pred_rows, journal)``. Each prediction row is
    ``{item, subject, original_gold, corrected_gold(=original), preds:{model->letter}}``
    on the same ``canonical_item_id`` as the corrections adapter, so the halves join.
    ``corrected_gold`` defaults to the source gold; ``stamp_predictions`` overrides it
    with the corrections decision when a corrections gold map is supplied.
    """
    fm = {**PRED_FIELD_MAP, **(field_map or {})}
    agg: Dict[str, Dict[str, Any]] = {}
    journal: List[Dict[str, Any]] = []

    for model, rows in samples_by_model.items():
        for s in rows:
            doc = s.get("doc") if isinstance(s.get("doc"), dict) else s
            q = _first(doc, fm["question"])
            choices = list(_first(doc, fm["choices"]) or [])
            ai = _first(doc, fm["answer"])
            if ai is None:
                ai = s.get("target")
            try:
                ai_int = int(ai)
            except (TypeError, ValueError):
                ai_int = -1
            subj = _first(doc, fm["subject"]) or s.get("subtask") or subject or "unknown"
            letter, rule = predicted_letter(s, len(choices), fm["resps"])
            item = canonical_item_id(subj, q, ai_int, n=id_len)
            og = [LETTERS[ai_int]] if 0 <= ai_int < 4 else []
            e = agg.setdefault(item, {"subject": subj, "original_gold": og, "preds": {}})
            if letter:
                e["preds"][model] = letter
            journal.append({"item": item, "model": model, "predicted": letter,
                            "rule": rule, "source_gold": (og[0] if og else None)})

    pred_rows = [{"item": k, "subject": v["subject"], "original_gold": v["original_gold"],
                  "corrected_gold": list(v["original_gold"]), "preds": v["preds"]}
                 for k, v in agg.items()]
    return pred_rows, journal


def _helm_correct_index(references: List[Dict[str, Any]]) -> int:
    for i, ref in enumerate(references):
        tags = ref.get("tags") or []
        if any(str(t).lower() == "correct" for t in tags):
            return i
    return -1


def _helm_predicted(completion_text: Optional[str],
                    references: List[Dict[str, Any]]) -> Tuple[Optional[str], str]:
    if not completion_text:
        return None, "empty completion"
    lt = _letter_from_text(completion_text)
    if lt:
        return lt, "letter from completion"
    ct = str(completion_text).strip().lower()
    for i, ref in enumerate(references):
        out = ref.get("output")
        txt = out.get("text") if isinstance(out, dict) else out
        if txt and str(txt).strip().lower() == ct and i < 4:
            return LETTERS[i], "matched reference text"
    return None, "unparseable completion"


def ingest_helm(scenario_states_by_model: Dict[str, Any], *, subject: Optional[str] = None,
                id_len: int = 80) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Merge per-model HELM ``scenario_state.json`` payloads into canonical prediction rows.

    HELM marks the gold with a CORRECT tag on a reference; the reference order is the
    option order, so the correct reference's index is the original answer index — the
    same key the MMLU-Redux corrections use, so the halves join on ``canonical_item_id``.
    The model's letter is read from the completion (a letter, or a match to a reference's
    text). ``subject`` is supplied per file (HELM splits a run per scenario/model).
    """
    agg: Dict[str, Dict[str, Any]] = {}
    journal: List[Dict[str, Any]] = []
    for model, payload in scenario_states_by_model.items():
        states = payload.get("request_states") if isinstance(payload, dict) else payload
        for rs in (states or []):
            inst = rs.get("instance") or {}
            inp = inst.get("input")
            q = inp.get("text") if isinstance(inp, dict) else inp
            refs = inst.get("references") or []
            ci = _helm_correct_index(refs)
            if ci < 0 or ci >= 4:
                journal.append({"item": None, "model": model, "predicted": None,
                                "rule": "no usable correct reference", "source_gold": None})
                continue
            og = LETTERS[ci]
            comps = (rs.get("result") or {}).get("completions") or []
            ctext = comps[0].get("text") if comps and isinstance(comps[0], dict) else None
            letter, rule = _helm_predicted(ctext, refs)
            subj = subject or inst.get("subject") or "unknown"
            item = canonical_item_id(subj, q, ci, n=id_len)
            e = agg.setdefault(item, {"subject": subj, "original_gold": [og], "preds": {}})
            if letter:
                e["preds"][model] = letter
            journal.append({"item": item, "model": model, "predicted": letter,
                            "rule": rule, "source_gold": og})
    pred_rows = [{"item": k, "subject": v["subject"], "original_gold": v["original_gold"],
                  "corrected_gold": list(v["original_gold"]), "preds": v["preds"]}
                 for k, v in agg.items()]
    return pred_rows, journal


INSPECT_FIELD_MAP = {
    "question": ["question", "query", "prompt", "input"],
    "choices": ["choices", "options"],
    "target": ["target", "answer"],
}


def _inspect_question(sample: Dict[str, Any], fm: Dict[str, List[str]]) -> Any:
    meta = sample.get("metadata") or {}
    for k in fm["question"]:
        if meta.get(k):
            return meta[k]
    inp = sample.get("input")
    if isinstance(inp, str):
        return inp
    if isinstance(inp, list):
        for m in reversed(inp):
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str):
                    return c
                if isinstance(c, list):
                    txt = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
                    if txt:
                        return txt
    return None


def _inspect_completion(sample: Dict[str, Any]) -> Optional[str]:
    out = sample.get("output")
    if isinstance(out, dict):
        if out.get("completion"):
            return out["completion"]
        ch = out.get("choices") or []
        if ch and isinstance(ch[0], dict):
            c = (ch[0].get("message") or {}).get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return None


def _target_to_letter(target: Any, choices: List[str]) -> Optional[str]:
    if isinstance(target, list):
        target = target[0] if target else None
    if target is None:
        return None
    s = str(target).strip()
    lt = _letter_from_text(s)
    if lt:
        return lt
    if re.fullmatch(r"[0-9]+", s):
        i = int(s)
        return LETTERS[i] if 0 <= i < 4 else None
    low = s.lower()
    for i, c in enumerate(choices):
        if c and str(c).strip().lower() == low and i < 4:
            return LETTERS[i]
    return None


def _inspect_predicted(sample: Dict[str, Any], choices: List[str]) -> Tuple[Optional[str], str]:
    scores = sample.get("scores")
    if isinstance(scores, dict):
        for sc in scores.values():
            ans = sc.get("answer") if isinstance(sc, dict) else None
            if ans:
                lt = _letter_from_text(str(ans))
                if lt:
                    return lt, "score answer (choice scorer)"
    comp = _inspect_completion(sample)
    if comp:
        lt = _letter_from_text(comp)
        if lt:
            return lt, "letter from completion"
        ct = str(comp).strip().lower()
        for i, c in enumerate(choices):
            if c and str(c).strip().lower() == ct and i < 4:
                return LETTERS[i], "matched choice text"
    return None, "unparseable output"


def ingest_inspect(logs_by_model: Dict[str, Any], *, subject: Optional[str] = None,
                   field_map: Optional[Dict[str, List[str]]] = None, id_len: int = 80
                   ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Merge per-model Inspect logs (EvalLog JSON) into canonical prediction rows.

    An Inspect sample carries the model's ``target`` (gold) and its answer (the choice
    scorer's ``answer``, or the completion). The gold letter's index is the answer index
    for ``canonical_item_id``. Because Inspect's ``input`` is a formatted prompt, the
    bare question is taken from ``metadata`` when present — keep it there (or set a
    ``field_map``) if you want these predictions to join MMLU-Redux corrections.

    Read ``.eval`` files with ``inspect log dump file.eval`` (JSON) or via the CLI, which
    lazily uses ``inspect_ai``. This function itself stays dependency-free: it parses dicts.
    """
    fm = {**INSPECT_FIELD_MAP, **(field_map or {})}
    agg: Dict[str, Dict[str, Any]] = {}
    journal: List[Dict[str, Any]] = []
    for model, log in logs_by_model.items():
        samples = log.get("samples") if isinstance(log, dict) else log
        for s in (samples or []):
            meta = s.get("metadata") or {}
            q = _inspect_question(s, fm)
            choices = list(s.get("choices") or meta.get("choices") or [])
            gold = _target_to_letter(s.get("target") if s.get("target") is not None
                                     else meta.get("target"), choices)
            if not gold:
                journal.append({"item": None, "model": model, "predicted": None,
                                "rule": "no usable target", "source_gold": None})
                continue
            gidx = LETTERS.index(gold)
            letter, rule = _inspect_predicted(s, choices)
            subj = subject or meta.get("subject") or "unknown"
            item = canonical_item_id(subj, q, gidx, n=id_len)
            e = agg.setdefault(item, {"subject": subj, "original_gold": [gold], "preds": {}})
            if letter:
                e["preds"][model] = letter
            journal.append({"item": item, "model": model, "predicted": letter,
                            "rule": rule, "source_gold": gold})
    pred_rows = [{"item": k, "subject": v["subject"], "original_gold": v["original_gold"],
                  "corrected_gold": list(v["original_gold"]), "preds": v["preds"]}
                 for k, v in agg.items()]
    return pred_rows, journal


def ingest_lm_eval_freeform(samples_by_model: Dict[str, Iterable[Dict[str, Any]]], *,
                            subject: Optional[str] = None,
                            field_map: Optional[Dict[str, List[str]]] = None,
                            id_len: int = 80) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Free-form predictions (e.g. GSM8K-Platinum): the model's parsed answer string.

    Reads the harness's filtered response as the answer, normalises it the same way the
    Platinum corrections normalise their targets, and joins on the free-form item id.
    """
    fm = {**PRED_FIELD_MAP, **(field_map or {})}
    agg: Dict[str, Dict[str, Any]] = {}
    journal: List[Dict[str, Any]] = []
    for model, rows in samples_by_model.items():
        for s in rows:
            doc = s.get("doc") if isinstance(s.get("doc"), dict) else s
            q = _first(doc, fm["question"])
            resp = _first(s, fm["resps"])
            while isinstance(resp, (list, tuple)) and resp:
                resp = resp[0]
            ans = normalize_answer(resp) or None
            tgt = _first(doc, fm["answer"])
            if tgt is None:
                tgt = s.get("target")
            og = normalize_answer(tgt)
            subj = _first(doc, fm["subject"]) or s.get("subtask") or subject or "unknown"
            item = canonical_item_id(subj, q, None, n=id_len)
            e = agg.setdefault(item, {"subject": subj, "original_gold": ([og] if og else []),
                                      "preds": {}})
            if ans:
                e["preds"][model] = ans
            journal.append({"item": item, "model": model, "predicted": ans,
                            "rule": "normalized free-form answer", "source_gold": og})
    pred_rows = [{"item": k, "subject": v["subject"], "original_gold": v["original_gold"],
                  "corrected_gold": list(v["original_gold"]), "preds": v["preds"]}
                 for k, v in agg.items()]
    return pred_rows, journal


def format_predictions_journal(journal: List[Dict[str, Any]], limit: Optional[int] = None,
                               source: str = "predictions") -> str:
    out = ["import journal · %s · %d (model, item) decodes" % (source, len(journal)), ""]
    rows = journal if limit is None else journal[:limit]
    for j in rows:
        out.append("%s" % j["item"])
        out.append("    %-18s predicted %r   (%s)" % (j["model"], j["predicted"], j["rule"]))
    if limit is not None and len(journal) > limit:
        out.append("… %d more" % (len(journal) - limit))
    return "\n".join(out)
