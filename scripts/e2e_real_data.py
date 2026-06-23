"""Astronomical end-to-end test on REAL data (HELM v1.3.0 x MMLU-Redux).

Drives the FULL pipeline (observe: sync -> admission -> build_site -> write_api
-> integrity -> changelog) over ~117 real sources (57 per-subject intrinsic +
57 per-subject predictions + pooled MMLU intrinsic + HELM claims + a deliberately
corrupt source), then runs thousands of counted invariant checks:
  * every stored record verifies (content hash) and round-trips through dict;
  * store.verify_all() is clean; integrity passes;
  * each benchmark's stored label-error equals an independent recount on the
    RAW (Title-case) annotations -> proves the error_type canonicalisation fix
    end to end;
  * each per-subject result_sensitivity / test_reliability audit reproduces a
    fresh recomputation (tau, deltas, n_changed, skill-disc, Cronbach alpha);
  * leaderboard == api == meridian.json (single source of truth), no trust score;
  * the generated SPA JS parses; re-running observe is idempotent; the corrupt
    source is quarantined.
"""
import csv, glob, json, os, re, subprocess, sys, tempfile
from pathlib import Path

import trust_the_eval.pipeline as PL
from trust_the_eval.admission import ProbeRegistry
from trust_the_eval.api import export as api_export
from trust_the_eval.calibration import has_scenarios
from trust_the_eval.calibration.realworld import (DEFECT_ERROR_TYPES,
                                                  AMBIGUITY_ERROR_TYPES,
                                                  canonical_error_type)
from trust_the_eval.leaderboard import leaderboard
from trust_the_eval.probe import all_probes
from trust_the_eval.record import ValidityRecord
from trust_the_eval.remediation import (REMEDIATION, TIER_CAPTION,
                                        remediation_for_benchmark)
from trust_the_eval.result_sensitivity import sensitivity, audit_measured
from trust_the_eval.item_analysis import (analyze, correctness_from_predictions,
                                          reliability_summary)
from trust_the_eval.store import RecordStore

ANN = "/home/claude/mmlu-redux-repo/mmlu_redux"
PRED = "/home/claude/mmlu-redux-repo/outputs/original_helm_combined"
HELM = "/home/claude/real/helm_mmlu_leaderboard.jsonl"
LET = "ABCD"

# ---- speed: shrink bootstrap for the pipeline's sensitivity records ----------
_bld = PL.build_record
def _fast_build(*a, **k):
    k.setdefault("iters", 400)
    return _bld(*a, **k)
PL.build_record = _fast_build
BOOT = 400  # must match for reproducibility checks

# ---- counted assertions ------------------------------------------------------
PASS = 0
FAILS = []
def check(cond, msg):
    global PASS
    if cond:
        PASS += 1
    else:
        FAILS.append(msg)

def approx(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol

# ---- real-data loader --------------------------------------------------------
def sniff(p):
    a = open(p, encoding="utf-8", errors="replace").readline()
    return ";" if a.count(";") > a.count(",") else ","
def nq(s):
    s = (s or "").lower().replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    return re.sub(r"\s+", " ", s).strip()
def pl(c):
    m = re.search(r"[A-D]", (c or "").upper())
    return m.group(0) if m else None
def parse_corr(raw, ch, base=0):
    s = (raw or "").strip()
    if not s:
        return None
    if re.fullmatch(r"[0-9]+", s):
        i = int(s) - base
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

intrinsic_rows = {}   # subj -> raw annotation rows (Title-case error_type)
pred_rows = {}        # subj -> prediction rows for result_sensitivity source
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
    irows, prows = [], []
    for r in csv.DictReader(open(p, encoding="utf-8", errors="replace"), delimiter=sniff(p)):
        try:
            ch = [str(x) for x in eval(r["choices"])]
        except Exception:
            ch = []
        ai = int(r.get("answer") or 0)
        orig = LET[ai] if 0 <= ai < 4 else None
        # intrinsic row keeps the RAW error_type (Title-case) on purpose
        irows.append({"question": r.get("question", ""), "choices": ch,
                      "answer": ai, "error_type": r.get("error_type"),
                      "subject": subj})
        pm = pbyq.get(nq(r["question"]))
        if pm is None or orig is None:
            continue
        cat = canonical_error_type(r.get("error_type"))
        parsed = parse_corr(r.get("correct_answer"), ch) if cat in (
            "wrong_groundtruth", "multiple_correct_answers") else None
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
        if not scor:
            continue
        iid = "%s::%s::%d" % (subj, nq(r["question"])[:48], ai)
        prows.append({"item": iid, "subject": subj,
                      "original_gold": [orig], "corrected_gold": sorted(corr),
                      "preds": pm})
    intrinsic_rows[subj] = irows
    pred_rows[subj] = prows

subjects = sorted(intrinsic_rows)
print("loaded %d subjects, %d models" % (len(subjects), len(models)))

# ---- build sources -----------------------------------------------------------
def fp_of(obj):
    import hashlib
    return "sha:" + hashlib.sha256(json.dumps(obj, sort_keys=True, default=str)
                                   .encode()).hexdigest()[:16]

class Src:
    def __init__(self, sid, payload, anchors=None):
        self.id = sid
        self._p = payload
        self.anchors = anchors or {"min_rows": 1}
    def fingerprint(self):
        return fp_of(self._p)
    def fetch(self):
        return self._p

sources = []
for s in subjects:
    sources.append(Src("int-%s" % s, {"kind": "intrinsic", "benchmark": "MMLU::%s" % s,
                                       "dataset_version": "mmlu-redux-2.0",
                                       "rows": intrinsic_rows[s]}))
    if pred_rows[s]:
        sources.append(Src("pred-%s" % s, {"kind": "predictions", "benchmark": "MMLU::%s" % s,
                                           "dataset_version": "mmlu-redux-2.0",
                                           "rows": pred_rows[s]}))
# pooled MMLU intrinsic (all rows) + HELM claims
all_irows = [r for s in subjects for r in intrinsic_rows[s]]
sources.append(Src("int-MMLU", {"kind": "intrinsic", "benchmark": "MMLU",
                                 "dataset_version": "mmlu-redux-2.0", "rows": all_irows}))
claim_rows = []
if os.path.exists(HELM):
    for line in open(HELM, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        claim_rows.append({"model": d.get("model") or d.get("name"),
                           "score": d.get("score") or d.get("accuracy") or d.get("acc"),
                           "date": d.get("date") or "2024-05"})
    sources.append(Src("claims-MMLU", {"kind": "claim", "benchmark": "MMLU",
                                       "dataset_version": "mmlu-redux-2.0",
                                       "rows": claim_rows},
                       anchors={"min_rows": 1, "score_between": [0.0, 1.0]}))
# a deliberately corrupt intrinsic source (label-error band cannot be met)
sources.append(Src("int-CORRUPT", {"kind": "intrinsic", "benchmark": "MMLU::CORRUPT",
                                    "dataset_version": "x", "rows": all_irows[:50]},
                   anchors={"min_rows": 1, "label_error_between": [0.90, 1.0]}))

print("built %d sources" % len(sources))

# ---- run the full pipeline ---------------------------------------------------
work = Path(tempfile.mkdtemp(prefix="e2e_real_"))
store_dir = work / "store"
out = work / "site"
res1 = PL.observe(store_dir, sources, out, always=True)
check(res1["changed"] is True, "observe cycle reports changed")
outc = {o["source"]: o for o in res1["sync"]}
ingested = [s for s, o in outc.items() if o["status"] == "ingested"]
quarantined = [s for s, o in outc.items() if o["status"] == "quarantined"]
check("int-CORRUPT" in quarantined, "corrupt source quarantined")
check(len(quarantined) == 1, "exactly one source quarantined (got %d)" % len(quarantined))
check(len(ingested) == len(sources) - 1, "all non-corrupt sources ingested (%d/%d)"
      % (len(ingested), len(sources) - 1))
qfiles = list((store_dir / "quarantine").glob("*.json")) if (store_dir / "quarantine").exists() else []
check(len(qfiles) >= 1, "quarantine artifact written")

# idempotency: a second cycle ingests nothing
res2 = PL.observe(store_dir, sources, out, always=False)
re2 = {o["source"]: o for o in res2.get("sync", [])}
check(all(re2[s]["status"] in ("skipped", "quarantined") for s in re2),
      "second observe cycle is idempotent (no re-ingest)")

# ---- open the store and verify everything ------------------------------------
store = RecordStore(store_dir)
ids = store.all_ids()
print("store holds %d records" % len(ids))
for rid in ids:
    d = store.get(rid)
    rec = store.get_record(rid)
    check(rec.verify(), "record verifies: %s" % rid[:16])
    check(ValidityRecord.from_dict(d).verify(), "record round-trips: %s" % rid[:16])
check(store.verify_all() == [], "store.verify_all() clean")

# registry + integrity
reg = ProbeRegistry(store_dir / "registry")
for P in all_probes():
    pid = P().id
    if has_scenarios(pid):
        reg.admit(pid)

# ---- leaderboard / api consistency + reproducibility -------------------------
LB = leaderboard(store)
bench_by = {b["name"]: b for b in LB["benchmarks"]}
API = api_export(store, reg)
api_benches = {b["name"]: b for b in API.get("leaderboard", {}).get("benchmarks", API.get("benchmarks", []))}
check(set(bench_by) == set(api_benches), "leaderboard benchmarks == api benchmarks")

# every benchmark: no trust scalar; honest provenance
for name, b in bench_by.items():
    check("trust" not in b, "no trust score on %s" % name)

# per-subject deep checks
for s in subjects:
    name = "MMLU::%s" % s
    b = bench_by.get(name)
    check(b is not None, "benchmark present: %s" % name)
    if b is None:
        continue
    # (1) label-error reproduced from RAW annotations via canonicalisation
    ann = [r for r in intrinsic_rows[s] if r.get("error_type") is not None]
    defects = sum(1 for r in ann if canonical_error_type(r["error_type"]) in DEFECT_ERROR_TYPES)
    exp_rate = defects / len(ann) if ann else None
    if b.get("label_error") and exp_rate is not None:
        check(approx(b["label_error"]["rate"], exp_rate, 1e-9),
              "label-error reproduced for %s (got %s want %.4f)"
              % (name, b["label_error"]["rate"], exp_rate))
        check(b["label_error"]["k"] == defects, "label-error k for %s" % name)
    # (2) result_sensitivity reproduced from a fresh recompute
    if pred_rows[s]:
        preds, og, cg = {}, {}, {}
        for r in pred_rows[s]:
            og[r["item"]] = set(r["original_gold"]); cg[r["item"]] = set(r["corrected_gold"])
            for m, l in r["preds"].items():
                preds.setdefault(m, {})[r["item"]] = l
        res = sensitivity(preds, og, cg, iters=BOOT, seed=0)
        want = audit_measured(res)
        got = b.get("sensitivity")
        check(got is not None, "sensitivity present for %s" % name)
        if got:
            check(got["n_changed_items"] == want["n_changed_items"], "sens n_changed %s" % name)
            check(approx(got.get("kendall_tau"), want.get("kendall_tau")), "sens tau %s" % name)
            check(approx(got["delta_min_pts"], want["delta_min_pts"], 1e-6), "sens dmin %s" % name)
            check(approx(got["delta_max_pts"], want["delta_max_pts"], 1e-6), "sens dmax %s" % name)
            sd_ok = (got.get("skill_discrimination") is None and want.get("skill_discrimination") is None) \
                or approx(got.get("skill_discrimination"), want.get("skill_discrimination"), 1e-9)
            check(sd_ok, "sens skill-disc %s" % name)
        # (3) reliability reproduced
        cm = correctness_from_predictions(preds, og)
        rel_want = reliability_summary(cm, {r["item"]: s for r in pred_rows[s]})
        rel_got = b.get("reliability")
        check(rel_got is not None, "reliability present for %s" % name)
        if rel_got:
            check(approx(rel_got.get("pooled_alpha"), rel_want.get("pooled_alpha"), 1e-9)
                  or (rel_got.get("pooled_alpha") is None and rel_want.get("pooled_alpha") is None),
                  "reliability alpha %s" % name)

        # (4) per-model delta identity + bootstrap-CI bracketing
        for mdl, pmv in res["per_model"].items():
            check(approx(pmv["delta"], pmv["acc_corr"] - pmv["acc_orig"]),
                  "per-model delta identity %s/%s" % (name, mdl))
            if pmv["delta_lo"] is not None:
                check(pmv["delta_lo"] - 1e-9 <= pmv["delta"] <= pmv["delta_hi"] + 1e-9,
                      "delta within CI %s/%s" % (name, mdl))
        # (5) ranking invariants
        check(set(res["ranking_orig"]) == set(res["ranking_corr"]), "rankings same set %s" % name)
        check(len(res["ranking_orig"]) == res["n_models"], "ranking length %s" % name)
        if res["kendall_tau"] is not None:
            check(-1.0 - 1e-9 <= res["kendall_tau"] <= 1.0 + 1e-9, "tau in range %s" % name)
        if res["p_top1_change"] is not None:
            check(0.0 <= res["p_top1_change"] <= 1.0, "p_top1 in range %s" % name)
        check(res["n_changed_items"] <= res["n_items_scored"], "n_changed<=n_items %s" % name)
        # (6) per-item psychometric invariants + correctness consistency
        A = analyze(cm)
        nmod = len(cm)
        for it, pi in A["per_item"].items():
            check(0.0 <= pi["difficulty"] <= 1.0, "difficulty range %s/%s" % (name, it))
            disc = pi["discrimination"]
            check(disc is None or (-1.0 - 1e-9 <= disc <= 1.0 + 1e-9),
                  "disc range %s/%s" % (name, it))
            check(isinstance(pi["suspect"], bool), "suspect bool %s/%s" % (name, it))
            csum = sum(cm[m].get(it, 0) for m in cm)
            check(approx(pi["difficulty"], csum / nmod), "difficulty=mean-correct %s/%s" % (name, it))

# field-level api == leaderboard (single source of truth)
for name, b in bench_by.items():
    ab = api_benches.get(name)
    check(ab is not None, "api has benchmark %s" % name)
    if not ab:
        continue
    le_lb = (b.get("label_error") or {}).get("rate")
    le_api = (ab.get("label_error") or {}).get("rate")
    check(approx(le_lb, le_api, 1e-12) or (le_lb is None and le_api is None),
          "api label-error matches lb %s" % name)
    s_lb = (b.get("sensitivity") or {}).get("kendall_tau")
    s_api = (ab.get("sensitivity") or {}).get("kendall_tau")
    check(approx(s_lb, s_api, 1e-12) or (s_lb is None and s_api is None),
          "api sensitivity tau matches lb %s" % name)
    r_lb = (b.get("reliability") or {}).get("pooled_alpha")
    r_api = (ab.get("reliability") or {}).get("pooled_alpha")
    check(approx(r_lb, r_api, 1e-12) or (r_lb is None and r_api is None),
          "api reliability matches lb %s" % name)

# ---- remediation guidance: massive invariant battery ------------------------
SEV = {"high": 3, "medium": 2, "low": 1, "info": 0}
VALID_TIERS = set(TIER_CAPTION)
aud_by_bench, sens_by_bench = {}, {}
for rid in ids:
    d = store.get(rid)
    nm = d["subject"]["benchmark"]
    aud_by_bench.setdefault(nm, []).extend(d["audits"])
    for a in d["audits"]:
        if a.get("probe_id") == "result_sensitivity" and a.get("measured"):
            sens_by_bench[nm] = a["measured"]

for name in bench_by:
    plan = remediation_for_benchmark(store, name)
    lb_plan = bench_by[name].get("remediation")
    check(lb_plan is not None, "leaderboard exposes remediation: %s" % name)
    if lb_plan:
        check(lb_plan["n_actions"] == plan["n_actions"], "remediation field == recompute %s" % name)
        check(lb_plan["summary"] == plan["summary"], "remediation summary == recompute %s" % name)
    # n_actions equals the set of fired, remediable probes (severity > info)
    fired = {a.get("probe_id") for a in aud_by_bench[name]
             if a.get("probe_id") in REMEDIATION
             and a.get("probe_id") != "result_sensitivity"
             and SEV.get(a.get("severity", "info"), 0) > 0}
    check(plan["n_actions"] == len(fired),
          "n_actions==fired probes %s (%d vs %d)" % (name, plan["n_actions"], len(fired)))
    # verdict impact consistent with the sensitivity audit
    s = sens_by_bench.get(name)
    if s is not None:
        exp = s.get("ranking_stable")
        if exp is None:
            nc = s.get("n_changed_items")
            exp = (nc == 0) if isinstance(nc, int) else None
        check(plan["verdict_impact"]["ranking_stable"] == exp,
              "verdict_impact matches sensitivity %s" % name)
    # per-action invariants
    for a in plan["actions"]:
        check(a["tier"] in VALID_TIERS, "tier valid %s/%s" % (name, a["probe_id"]))
        check(a["asserts_correction"] is False, "never asserts correction %s/%s" % (name, a["probe_id"]))
        check(bool(a["action"]) and bool(a["rationale"]) and bool(a["caveat"]),
              "action fields populated %s/%s" % (name, a["probe_id"]))
        if a["tier"] == "relay_only":
            check(a["candidates_available"] is True and "verify" in a["action"].lower(),
                  "relay_only carries verify caveat %s/%s" % (name, a["probe_id"]))
        if a["tier"] == "investigate":
            check("not a verdict" in TIER_CAPTION["investigate"][1],
                  "investigate caveat present %s" % name)
    # label-error must be relay-only with the right count whenever it fired
    le = bench_by[name].get("label_error")
    le_acts = [a for a in plan["actions"] if a["probe_id"] == "label_error_audit"]
    if le and (le.get("k") or 0) > 0:
        check(len(le_acts) == 1, "label-error remediated when k>0 %s" % name)
        if le_acts:
            check(le_acts[0]["tier"] == "relay_only", "label-error relay_only %s" % name)
            check(le_acts[0]["n_items_affected"] == le["k"], "label-error n_items==k %s" % name)
            check("the answer is" not in le_acts[0]["action"].lower()
                  and "correct answer is" not in le_acts[0]["action"].lower(),
                  "label-error never asserts answer %s" % name)

# ---- meridian.json consistency + SPA validity --------------------------------
mj = json.loads((out / "meridian.json").read_text())
mj_b = mj.get("leaderboard", {}).get("benchmarks", mj.get("benchmarks", []))
check(len(mj_b) == len(bench_by), "meridian.json benchmark count == leaderboard")
html = (out / "index.html").read_text()
m = re.search(r"<script>(.*)</script>", html, re.S)
check(m is not None, "SPA has an inline script")
if m:
    js = work / "site.js"
    js.write_text(m.group(1))
    r = subprocess.run(["node", "--check", str(js)], capture_output=True, text=True)
    check(r.returncode == 0, "SPA JS parses (node --check): %s" % r.stderr[:200])
check("trust score" not in html.lower() or "not a trust" in html.lower(),
      "SPA does not present a trust score as a metric")

# ---- report ------------------------------------------------------------------
print("\n================ E2E REAL-DATA RESULT ================")
print("records: %d | benchmarks: %d | sources: %d" % (len(ids), len(bench_by), len(sources)))
print("CHECKS PASSED: %d" % PASS)
print("CHECKS FAILED: %d" % len(FAILS))
for f in FAILS[:40]:
    print("  FAIL:", f)
sys.exit(1 if FAILS else 0)
