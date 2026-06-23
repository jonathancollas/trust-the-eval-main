"""Build the navigable validity observatory from REAL data (HELM v1.3.0 x MMLU-Redux).

Loads the per-subject annotations and 10-model HELM predictions, runs them through the
pipeline (`sync`), then renders the standalone observatory via
`trust_the_eval.observatory_ui`. Mirrors scripts/e2e_real_data.py for data location.

    python scripts/build_observatory_ui.py [DATA_ROOT] [OUT_HTML]

DATA_ROOT defaults to ~/mmlu-redux-repo (clone of github.com/aryopg/mmlu-redux);
OUT_HTML defaults to /mnt/user-data/outputs/meridian-ihm.html. Every figure rendered
reconstructs from its trace (see tests/test_observatory_ui.py).
"""
import csv
import glob
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import trust_the_eval.pipeline as PL
from trust_the_eval.calibration.realworld import canonical_error_type
from trust_the_eval.observatory_ui import assemble_ui_data, build_ui_html
from trust_the_eval.state import SourceState
from trust_the_eval.store import RecordStore

DATA_ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/mmlu-redux-repo")
OUT_HTML = sys.argv[2] if len(sys.argv) > 2 else "/mnt/user-data/outputs/meridian-ihm.html"
ANN = os.path.join(DATA_ROOT, "mmlu_redux")
PRED = os.path.join(DATA_ROOT, "outputs", "original_helm_combined")
HELM = os.path.expanduser("~/real/helm_mmlu_leaderboard.jsonl")
LET = "ABCD"
SPOT = ["MMLU", "MMLU::virology", "MMLU::college_chemistry", "MMLU::high_school_geography"]

# fast bootstrap for the pipeline's sensitivity records (point estimates unaffected)
_bld = PL.build_record
PL.build_record = lambda *a, **k: (k.setdefault("iters", 400), _bld(*a, **k))[1]


def sniff(p):
    a = open(p, encoding="utf-8", errors="replace").readline()
    return ";" if a.count(";") > a.count(",") else ","


def nq(s):
    s = (s or "").lower().replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", s.replace("\u2019", "'").replace("\u2018", "'")).strip()


def pl(c):
    m = re.search(r"[A-D]", (c or "").upper())
    return m.group(0) if m else None


def parse_corr(raw, ch):
    s = (raw or "").strip()
    if not s:
        return None
    if re.fullmatch(r"[0-9]+", s):
        i = int(s)
        return {LET[i]} if 0 <= i < 4 else None
    low = s.lower()
    for i, c in enumerate(ch):
        if c and c.strip().lower() == low:
            return {LET[i]}
    hit = [i for i, c in enumerate(ch) if c and c.strip().lower() in low]
    if len(hit) == 1:
        return {LET[hit[0]]}
    L = set(re.findall(r"(?<![A-Z])([A-D])(?![A-Z])", s.upper()))
    return L or None


class Src:
    def __init__(self, sid, payload, anchors=None):
        self.id = sid
        self._p = payload
        self.anchors = anchors or {"min_rows": 1}

    def fingerprint(self):
        return "fp-" + self.id

    def fetch(self):
        return self._p


def main():
    intrinsic_rows, pred_rows, cand_items = {}, {}, {}
    models = None
    for p in sorted(glob.glob(os.path.join(ANN, "mmlu_*.csv"))):
        subj = os.path.basename(p)[5:-4]
        pp = os.path.join(PRED, subj + ".csv")
        if not os.path.exists(pp):
            continue
        pr = csv.DictReader(open(pp, encoding="utf-8", errors="replace"), delimiter=sniff(pp))
        mc = [c for c in pr.fieldnames if c not in ("question", "choices")]
        models = models or mc
        pbyq = {nq(r["question"]): {m: pl(r[m]) for m in mc} for r in pr}
        bench = "MMLU::%s" % subj
        irows, prows, cands = [], [], []
        for r in csv.DictReader(open(p, encoding="utf-8", errors="replace"), delimiter=sniff(p)):
            try:
                ch = [str(x) for x in eval(r["choices"])]
            except Exception:
                ch = []
            ai = int(r.get("answer") or 0)
            orig = LET[ai] if 0 <= ai < 4 else None
            irows.append({"question": r.get("question", ""), "choices": ch, "answer": ai,
                          "error_type": r.get("error_type"), "subject": subj})
            pm = pbyq.get(nq(r["question"]))
            if pm is None or orig is None:
                continue
            cat = canonical_error_type(r.get("error_type"))
            parsed = parse_corr(r.get("correct_answer"), ch) if cat in ("wrong_groundtruth", "multiple_correct_answers") else None
            if cat in (None, "ok", "bad_question_clarity", "bad_options_clarity", "expert"):
                corr, scor = {orig}, True
            elif cat == "wrong_groundtruth":
                corr, scor = (parsed, True) if parsed else ({orig}, False)
            elif cat == "multiple_correct_answers":
                corr, scor = ({orig} | (parsed or set())), True
            elif cat == "no_correct_answer":
                corr, scor = {orig}, False
            else:
                corr, scor = {orig}, True
            if cat == "wrong_groundtruth" and parsed and len(cands) < 6:
                cands.append({"q": r.get("question", "")[:120], "orig": orig,
                              "cand": sorted(corr), "etype": r.get("error_type")})
            if not scor:
                continue
            iid = "%s::%s::%d" % (subj, nq(r["question"])[:40], ai)
            prows.append({"item": iid, "subject": subj, "original_gold": [orig],
                          "corrected_gold": sorted(corr), "preds": pm})
        intrinsic_rows[bench] = irows
        pred_rows[bench] = prows
        cand_items[subj] = cands

    subjects = sorted(b.split("::")[1] for b in intrinsic_rows)
    # pooled MMLU
    intrinsic_rows["MMLU"] = [r for b in list(intrinsic_rows) for r in intrinsic_rows[b]]
    pred_rows["MMLU"] = [r for b in list(pred_rows) for r in pred_rows[b]]

    sources = []
    for subj in subjects:
        b = "MMLU::%s" % subj
        sources.append(Src("int-%s" % subj, {"kind": "intrinsic", "benchmark": b,
                                              "dataset_version": "mmlu-redux-2.0", "rows": intrinsic_rows[b]}))
        if pred_rows[b]:
            sources.append(Src("pred-%s" % subj, {"kind": "predictions", "benchmark": b,
                                                  "dataset_version": "mmlu-redux-2.0", "rows": pred_rows[b]}))
    sources.append(Src("int-MMLU", {"kind": "intrinsic", "benchmark": "MMLU",
                                    "dataset_version": "mmlu-redux-2.0", "rows": intrinsic_rows["MMLU"]}))
    sources.append(Src("pred-MMLU", {"kind": "predictions", "benchmark": "MMLU",
                                     "dataset_version": "mmlu-redux-2.0", "rows": pred_rows["MMLU"]}))
    if os.path.exists(HELM):
        claims = []
        for line in open(HELM, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                claims.append({"model": d.get("model") or d.get("name"),
                               "score": d.get("score") or d.get("accuracy") or d.get("acc")})
        sources.append(Src("claims-MMLU", {"kind": "claim", "benchmark": "MMLU",
                                           "dataset_version": "mmlu-redux-2.0", "rows": claims},
                           anchors={"min_rows": 1, "score_between": [0.0, 1.0]}))

    root = Path(tempfile.mkdtemp(prefix="obsui_"))
    store = RecordStore(root / "s")
    PL.sync(store, sources, SourceState(root / "s"), root=root / "s",
            calibrations_dataset={}, calibrations_result={})

    data = assemble_ui_data(store, pred_rows, intrinsic_rows, SPOT, models, candidates=cand_items)
    html = build_ui_html(data)
    Path(OUT_HTML).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_HTML).write_text(html, encoding="utf-8")
    print("wrote %s: %.0f KB | %d benchmarks | %d models | spotlight=%s"
          % (OUT_HTML, len(html) / 1024, data["n_benchmarks"], len(data["models"]),
             [n for n in SPOT if n in data["spotlight"]]))


if __name__ == "__main__":
    main()
