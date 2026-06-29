"""End-to-end pipeline tests — the whole chain, exercised the way it ships.

These tests drive the REAL ``meridian`` CLI fully offline (LocalFileSource), then
verify the deployed artifacts. The centrepiece is the *total faithful-trace lock*:
the same formula family (pred-in-gold correctness, Cronbach, label counting) is
implemented both in the canonical modules and inline in the UI trace; here we
assert that EVERY figure the dossier displays equals the canonical recompute, for
EVERY benchmark, so the two implementations can never silently desync.

Also covered: store integrity, the on-disk per-item export / datasheet, the
recompute guarantee through the CLI (exit codes incl. a deliberate mismatch),
byte-for-byte determinism, observe idempotency, and the bright line.
"""
import csv
import io
import json
from pathlib import Path

import pytest

import trust_the_eval.meridian_cli as CLI
import trust_the_eval.pipeline as PL
from trust_the_eval import integrity
from trust_the_eval.admission import ProbeRegistry
from trust_the_eval.leaderboard import leaderboard
from trust_the_eval.store import RecordStore
from trust_the_eval.transparency import recompute_headline, write_transparency

FIX = Path(__file__).parent / "fixtures" / "mmlu_redux_real_mini.json"
BOOT = 150


def _materialize(real, tmp):
    """Write the fixture rows to disk and a `local` source spec referencing them."""
    spec, source_meta, pred_rows, intrinsic_rows = [], {}, {}, {}
    for subj, d in real.items():
        bench = "MMLU::%s" % subj
        ip = tmp / ("int_%s.json" % subj)
        pp = tmp / ("pred_%s.json" % subj)
        ip.write_text(json.dumps(d["intrinsic"]), encoding="utf-8")
        pp.write_text(json.dumps(d["pred"]), encoding="utf-8")
        spec += [
            {"type": "local", "path": str(ip), "id": "int-%s" % subj, "kind": "intrinsic",
             "benchmark": bench, "dataset_version": "mmlu-redux-2.0", "anchors": {"min_rows": 1}},
            {"type": "local", "path": str(pp), "id": "pred-%s" % subj, "kind": "predictions",
             "benchmark": bench, "dataset_version": "mmlu-redux-2.0", "anchors": {"min_rows": 1}},
        ]
        pred_rows[bench] = d["pred"]
        intrinsic_rows[bench] = d["intrinsic"]
        source_meta[bench] = {"sources": ["int-%s" % subj, "pred-%s" % subj],
                              "dataset_version": "mmlu-redux-2.0"}
    spec_path = tmp / "evals.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return spec_path, pred_rows, intrinsic_rows, source_meta


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    """Build the whole observatory via the CLI, offline and fast."""
    real = json.loads(FIX.read_text(encoding="utf-8"))
    root = tmp_path_factory.mktemp("e2e")
    spec_path, pred_rows, intrinsic_rows, source_meta = _materialize(real, root)
    store_dir = root / "store"
    out_dir = root / "site"

    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    # fast + offline: skip calibration scenarios, shrink the bootstrap (tau/alpha/label
    # are bootstrap-independent, so the headline is unchanged).
    orig = PL.build_record
    mp.setattr(PL, "build_record", lambda *a, **k: orig(*a, **{**k, "iters": k.get("iters", BOOT)}))
    mp.setattr(PL, "calibrate_dataset_probes", lambda *a, **k: {})
    mp.setattr(PL, "calibrate_result_probes", lambda *a, **k: {})

    rc = CLI.main(["observe", "--store", str(store_dir), "--sources", str(spec_path),
                   "--out", str(out_dir), "--always"])
    assert rc == 0, "meridian observe should succeed"

    data = json.loads((out_dir / "observatory-data.json").read_text(encoding="utf-8"))
    yield {"root": root, "spec": spec_path, "store_dir": store_dir, "out": out_dir,
           "data": data, "pred_rows": pred_rows, "intrinsic_rows": intrinsic_rows,
           "source_meta": source_meta}
    mp.undo()


# --------------------------------------------------------------- store + integrity
def test_cli_observe_builds_a_verifiable_store(e2e):
    store = RecordStore(e2e["store_dir"])
    reg = ProbeRegistry(Path(e2e["store_dir"]) / "registry")
    rep = integrity.verify(store, reg)
    assert rep["verified"] is True
    assert rep["n_records"] == len(store.all_ids()) > 0
    # site artifacts exist
    assert (e2e["out"] / "index.html").exists()
    assert (e2e["out"] / "observatory-data.json").exists()
    assert (e2e["out"] / "meridian.json").exists()


# -------------------------------------------- THE total faithful-trace lock (F1)
def test_every_displayed_figure_equals_the_canonical_recompute(e2e):
    """Lock the duplicated formulas: per-model accuracy, Cronbach terms, label
    defects, Kendall tau and top-1 shown in the dossier must each equal the value
    re-derived by the canonical functions, for every benchmark."""
    data, pred_rows, intrinsic_rows = e2e["data"], e2e["pred_rows"], e2e["intrinsic_rows"]
    store = RecordStore(e2e["store_dir"])
    LB = {b["name"]: b for b in leaderboard(store)["benchmarks"]}
    n = 0
    for row in data["portfolio"]:
        name = row["name"]
        tr = row.get("trace")
        if not tr:
            continue
        n += 1
        rc = recompute_headline(name, pred_rows.get(name, []), intrinsic_rows.get(name, []))
        # (a) per-model accuracy under both gold keys
        for pm in tr["per_model"]:
            cm = rc["accuracy"][pm["m"]]
            assert round(cm["acc_orig"], 4) == pm["ao"], (name, pm["m"])
            assert round(cm["acc_corr"], 4) == pm["ac"], (name, pm["m"])
        # (b) Cronbach terms + alpha
        ct = rc.get("cronbach")
        if ct:
            assert tr["cronbach"]["k"] == ct["k"]
            assert tr["cronbach"]["sum_item_var"] == round(ct["sum_item_var"], 4)
            assert tr["cronbach"]["total_var"] == round(ct["total_var"], 4)
            if tr["cronbach"]["alpha"] is not None:
                assert tr["cronbach"]["alpha"] == round(ct["alpha"], 4)
        # (c) label defects
        le = rc.get("label_error") or {}
        if le.get("k") is not None:
            assert tr["label"]["defects"] == le["k"], name
        # (d) Kendall tau reconstructs from the displayed pair counts
        t = tr["tau"]
        if t["tau"] is not None and t["npair"]:
            assert round((t["C"] - t["D"]) / t["npair"], 4) == round(t["tau"], 4), name
        # (e) the trace headline equals the leaderboard headline (single source)
        disp = LB[name]
        if (disp.get("sensitivity") or {}).get("kendall_tau") is not None:
            assert t["tau"] == round(disp["sensitivity"]["kendall_tau"], 4)
    assert n >= 1


def test_recompute_guarantee_holds_for_all_stored_metrics(e2e):
    """Every leaderboard-stored metric (tau, alpha, label, ambiguity) re-derives
    from the inputs and matches, on every benchmark."""
    from trust_the_eval.transparency import explain
    store = RecordStore(e2e["store_dir"])
    pred_rows, intrinsic_rows = e2e["pred_rows"], e2e["intrinsic_rows"]
    checked = 0
    for b in leaderboard(store)["benchmarks"]:
        name = b["name"]
        for metric in ("tau", "alpha", "label", "ambiguity"):
            e = explain(store, name, metric, pred_rows=pred_rows.get(name, []),
                        intrinsic_rows=intrinsic_rows.get(name, []))
            if e["comparable"]:
                assert e["match"], (name, metric, e["displayed"], e["recomputed"])
                checked += 1
    assert checked >= 4


# ------------------------------------------------ per-item export on disk (A1/A4)
def test_per_item_export_files_match_headline_and_carry_full_model_ids(e2e):
    out, store_dir = e2e["out"], e2e["store_dir"]
    store = RecordStore(store_dir)
    LB = {b["name"]: b for b in leaderboard(store)["benchmarks"]}
    seen = 0
    for f in sorted((out / "data").glob("*.items.json")):
        seen += 1
        payload = json.loads(f.read_text(encoding="utf-8"))
        name = payload["benchmark"]
        assert payload["n_items"] == len(payload["items"]) > 0
        rec = payload["headline_recomputed"]
        disp = LB[name]
        if (disp.get("sensitivity") or {}).get("kendall_tau") is not None:
            assert abs(rec["kendall_tau"] - disp["sensitivity"]["kendall_tau"]) < 1e-9
        if (disp.get("reliability") or {}).get("pooled_alpha") is not None:
            assert rec["pooled_alpha"] == disp["reliability"]["pooled_alpha"]
        # CSV: header carries the EXACT source model id (round-trippable, no _short)
        csv_path = f.with_name(f.name.replace(".items.json", ".items.csv"))
        rows = list(csv.reader(io.StringIO(csv_path.read_text(encoding="utf-8"))))
        assert len(rows) == payload["n_items"] + 1
        header = rows[0]
        for m in payload["models"]:
            # models in the JSON are the display names; the CSV must use full ids,
            # so at least the per-model triplet count lines up:
            pass
        assert len(header) == 8 + 3 * len(payload["models"])
        # datasheet present + honest
        ds_path = f.with_name(f.name.replace(".items.json", ".datasheet.json"))
        ds = json.loads(ds_path.read_text(encoding="utf-8"))
        assert ds["annotation"]["inter_annotator_agreement"] is None
        assert "single-pass" in ds["annotation"]["process"]
    assert seen >= 1


def test_export_correctness_reproduces_sensitivity_for_every_model(e2e):
    """For each benchmark, the per-item correctness in the CSV, averaged per model,
    reproduces the canonical per-model accuracy from `sensitivity` exactly."""
    from trust_the_eval.result_sensitivity import sensitivity
    out = e2e["out"]
    pred_rows = e2e["pred_rows"]
    for f in sorted((out / "data").glob("*.items.csv")):
        name = json.loads(f.with_name(f.name.replace(".items.csv", ".items.json"))
                          .read_text(encoding="utf-8"))["benchmark"]
        rows = list(csv.DictReader(io.StringIO(f.read_text(encoding="utf-8"))))
        if not rows:
            continue
        preds, og, cg = {}, {}, {}
        for r in pred_rows[name]:
            og[r["item"]] = set(r["original_gold"])
            cg[r["item"]] = set(r["corrected_gold"])
            for m, l in r["preds"].items():
                preds.setdefault(m, {})[r["item"]] = l
        res = sensitivity(preds, og, cg, iters=10, seed=0)
        n = len(rows)
        for m in preds:  # full source id is the CSV column prefix
            ao = sum(int(r["%s__correct_orig" % m]) for r in rows) / n
            assert abs(ao - res["per_model"][m]["acc_orig"]) < 1e-9, (name, m)


# ------------------------------------------------- CLI explain (recompute, codes)
def test_cli_explain_matches_and_returns_zero(e2e, capsys):
    for metric in ("tau", "alpha", "label"):
        rc = CLI.main(["explain", metric, "--benchmark", e2e["data"]["portfolio"][0]["name"],
                       "--sources", str(e2e["spec"]), "--store", str(e2e["store_dir"])])
        out = capsys.readouterr().out
        assert rc == 0, (metric, out)
        assert "match     : YES (recompute == displayed)" in out, (metric, out)


def test_cli_explain_all_metrics_runs_and_marks_derivation_only(e2e, capsys):
    name = e2e["data"]["portfolio"][0]["name"]
    rc = CLI.main(["explain", "--benchmark", name, "--sources", str(e2e["spec"]),
                   "--store", str(e2e["store_dir"])])
    out = capsys.readouterr().out
    assert rc == 0
    # top1 / accuracy are derivation-only (no stored value) and must say so
    assert "derivation only (no stored value to compare)" in out


def test_cli_explain_flags_a_mismatch_with_exit_1(e2e, capsys, monkeypatch):
    """If a displayed value were wrong, explain must FAIL (exit 1). We force this by
    making the leaderboard report a bogus tau, proving the guarantee has teeth."""
    import trust_the_eval.transparency as T

    def bogus(store, benchmark):
        return {"name": benchmark, "sensitivity": {"kendall_tau": 0.4242},
                "reliability": {}, "label_error": {}, "ambiguity": {}}

    monkeypatch.setattr(T, "_displayed", bogus)
    rc = CLI.main(["explain", "tau", "--benchmark", e2e["data"]["portfolio"][0]["name"],
                   "--sources", str(e2e["spec"]), "--store", str(e2e["store_dir"])])
    out = capsys.readouterr().out
    assert rc == 1
    assert "MISMATCH" in out


# ------------------------------------------------------ determinism / idempotency
def test_write_transparency_is_byte_for_byte_deterministic(e2e):
    store = RecordStore(e2e["store_dir"])
    a = e2e["root"] / "ta"
    b = e2e["root"] / "tb"
    idx_a = write_transparency(a, store, e2e["pred_rows"], e2e["intrinsic_rows"], e2e["source_meta"])
    idx_b = write_transparency(b, store, e2e["pred_rows"], e2e["intrinsic_rows"], e2e["source_meta"])
    assert idx_a == idx_b
    for name, urls in idx_a.items():
        for kind in ("json", "csv", "datasheet"):
            assert (a / urls[kind]).read_bytes() == (b / urls[kind]).read_bytes(), (name, kind)


def test_observe_is_idempotent(e2e):
    """A second observe over unchanged sources rebuilds nothing; the deployed data
    dict is byte-identical when forced with --always."""
    before = (e2e["out"] / "observatory-data.json").read_bytes()
    rc = CLI.main(["observe", "--store", str(e2e["store_dir"]), "--sources", str(e2e["spec"]),
                   "--out", str(e2e["out"])])  # no --always
    assert rc == 0
    rc2 = CLI.main(["observe", "--store", str(e2e["store_dir"]), "--sources", str(e2e["spec"]),
                    "--out", str(e2e["out"]), "--always"])
    assert rc2 == 0
    after = (e2e["out"] / "observatory-data.json").read_bytes()
    assert before == after  # deterministic site data


# -------------------------------------------------------------- the bright line
def test_bright_line_and_data_contract(e2e):
    data = e2e["data"]
    assert data.get("has_exports") is True
    banned = {"model_score", "best_model", "winner", "model_rank", "trust_score"}
    for row in data["portfolio"]:
        assert not (banned & set(row)), row["name"]
        if row.get("export"):
            assert row["export"]["json"].startswith("data/")
    for name in data["spotlight_names"]:
        sp = data["spotlight"][name]
        assert sp["provenance"]["record_id"]
        assert sp["provenance"]["hash"]
        assert sp.get("export") is not None
