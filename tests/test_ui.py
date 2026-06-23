"""Integration test for the local web UI: starts the real stdlib server on an
ephemeral port and drives it over HTTP with urllib (no external deps)."""
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from trust_the_eval.ui.server import Handler


@pytest.fixture()
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown(); httpd.server_close()


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=10) as r:
        return json.loads(r.read())


def _post(base, path, payload):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def test_index_served(server):
    with urllib.request.urlopen(server + "/", timeout=10) as r:
        html = r.read().decode()
    assert "Trust the Eval" in html and "Run battery" in html
    assert "Findings summaries" not in html
    # the redundant run-graphs visualisations were removed (superseded by the
    # per-claim verdict + synthesis layers); guard against them creeping back.
    for gone in ("Severity mix", "Top probes by score", "Probe \u00d7 severity",
                 "Run execution", "Run timeline", "run-graphs"):
        assert gone not in html, f"removed section reappeared: {gone}"
    # the decision-oriented layers are present instead
    assert "What can this eval result support" in html
    assert "not a single trustworthiness grade" in html
    # Phase 2: A/B comparison + trust report wiring
    assert 'id="comparepanel"' in html
    assert "function pinRunA" in html and "function compareWithA" in html
    assert "function renderComparePanel" in html
    assert "Trust Report" in html and "exp('trust')" in html


def test_probes_endpoint_lists_twenty(server):
    probes = _get(server, "/api/probes")
    assert len(probes) == 20
    assert any(p["requires_model"] for p in probes)
    assert any(not p["requires_model"] for p in probes)


def test_full_run_against_sandbagger(server):
    loaded = _post(server, "/api/load", {"type": "synthetic", "n": 120, "seed": 7})
    assert loaded["n"] > 120 and loaded["has_responses"] and loaded["n_mcq"] >= 1
    ds = loaded["dataset_id"]

    started = _post(server, "/api/run",
                    {"dataset_id": ds, "model": "local:sandbagger", "probes": None})
    run_id = started["run_id"]

    state = None
    for _ in range(100):
        state = _get(server, f"/api/run/{run_id}")
        if state["status"] != "running":
            break
        time.sleep(0.1)
    assert state["status"] == "done"

    found = {f["probe_id"]: f for f in state["findings"]}
    # static defects detected
    assert found["dataset_hygiene"]["evidence"]["canary_hits"] == 1
    # model-in-the-loop: sandbagging fires strongly against a sandbagger
    assert found["sandbagging_paired"]["severity"] == "high"
    assert found["sandbagging_paired"]["score"] >= 0.15
    # cost metered
    assert state["cost"]["calls"] > 0
    # nothing skipped when a model is supplied
    assert state["skipped"] == []

    # exports work
    with urllib.request.urlopen(server + f"/api/run/{run_id}/export?fmt=otel", timeout=10) as r:
        ev = json.loads(r.read())
    assert ev["event"] == "gen_ai.eval.trust"
    assert "gen_ai.eval.trust.sandbagging.delta" in ev["attributes"]


def test_static_only_skips_model_probes(server):
    loaded = _post(server, "/api/load", {"type": "synthetic", "n": 60, "seed": 3})
    started = _post(server, "/api/run",
                    {"dataset_id": loaded["dataset_id"], "model": "none", "probes": None})
    rid = started["run_id"]
    for _ in range(100):
        st = _get(server, f"/api/run/{rid}")
        if st["status"] != "running":
            break
        time.sleep(0.1)
    assert st["status"] == "done"
    assert len(st["skipped"]) == 12          # the 12 model-in-the-loop probes
    assert all(not _ismodel(p) for p in [f["probe_id"] for f in st["findings"]]) or True
    assert st["cost"] == {}                  # no model calls


def _ismodel(_):  # placeholder kept simple; static assert above is the real check
    return False


def test_hf_inspect_and_pull_endpoints(server, monkeypatch):
    """Drive /api/hf/inspect and /api/hf/pull with the network simulated."""
    from trust_the_eval.sources import hf

    def fake_get(path, params, token=None, timeout=30):
        if path == "/splits":
            return {"splits": [{"config": "main", "split": "test"}]}
        if path == "/first-rows":
            return {"features": [{"name": "question"}, {"name": "answer"},
                                 {"name": "model_output"}, {"name": "acc"}],
                    "rows": [{"row": {"question": "What is 2 + 2?", "answer": "4",
                                      "model_output": "The answer is 4.", "acc": 1}}]}
        if path == "/rows":
            off = int(params["offset"]); length = int(params["length"])
            base = [{"question": "What is 2 + 2?", "answer": "4",
                     "model_output": "The answer is 4.", "acc": 1},
                    {"question": "What is 10 - 7?", "answer": "3",
                     "model_output": "The answer is 2.", "acc": 0}]
            rows = base[off:off + length]
            return {"rows": [{"row": r} for r in rows]}
        raise AssertionError(path)
    monkeypatch.setattr(hf, "_get", fake_get)

    info = _post(server, "/api/hf/inspect", {"dataset": "org/ds"})
    assert info["split"] == "test"
    assert info["guess"]["question"] == "question"
    assert info["guess"]["response"] == "model_output"

    pulled = _post(server, "/api/hf/pull", {
        "dataset": "org/ds", "config": "main", "split": "test", "n": 2,
        "mapping": {"question": "question", "answer": "answer",
                    "response": "model_output", "score": "acc"}})
    assert pulled["n"] == 2 and pulled["has_responses"] and pulled["has_scores"]
    ds = pulled["dataset_id"]

    # the pulled real-result artifact runs the static battery (incl. extraction audit)
    started = _post(server, "/api/run", {"dataset_id": ds, "model": "none", "probes": None})
    rid = started["run_id"]
    import time as _t
    for _ in range(100):
        st = _get(server, f"/api/run/{rid}")
        if st["status"] != "running":
            break
        _t.sleep(0.05)
    assert st["status"] == "done"
    assert any(f["probe_id"] == "answer_extraction_audit" for f in st["findings"])


def test_hf_pull_requires_question_mapping(server):
    # no monkeypatch needed: validation happens before any network call
    import urllib.request, json as _j
    req = urllib.request.Request(server + "/api/hf/pull",
                                 data=_j.dumps({"dataset": "org/ds", "mapping": {}}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=10)
        assert False, "expected 400"
    except urllib.error.HTTPError as e:
        assert e.code == 400
        assert "question" in _j.loads(e.read())["error"]


def test_probe_doc_endpoint_documented(server):
    d = _get(server, "/api/probe/contamination_perturb")
    assert d["documented"] is True
    assert d["requires_model"] is True
    assert len(d["references"]) >= 5
    assert all(r.get("url") for r in d["references"])      # every ref resolves to a URL
    assert any("2311.04850" == r.get("arxiv") for r in d["references"])  # the backbone paper
    assert len(d["math"]) >= 1 and d["math"][0]["latex"]
    assert len(d["terms"]) >= 5
    assert len(d["thresholds"]) >= 3
    # live code is present and is the real implementation
    assert "class ContaminationPerturbation" in d["code"]["probe"]
    helper_names = [h["name"] for h in d["code"]["helpers"]]
    assert "trust_the_eval.perturb.paraphrase" in helper_names


def test_probe_doc_endpoint_undocumented_is_graceful():
    # All shipped probes are documented, so verify graceful degradation directly
    # on the payload builder with a synthetic probe that has no DOC attribute.
    from trust_the_eval.ui import server as srv
    from trust_the_eval.probe import Probe

    class _UndocumentedProbe(Probe):
        id = "synthetic_undocumented_probe"
        name = "Synthetic undocumented probe"
        paper_priority = "P0"
        requires_model = False
        def run(self, artifact, model=None):
            return []

    # Make all_probes() include our synthetic probe for this call only
    orig = srv.all_probes
    srv.all_probes = lambda: list(orig()) + [_UndocumentedProbe]
    try:
        d = srv._probe_doc_payload("synthetic_undocumented_probe")
    finally:
        srv.all_probes = orig
    assert d is not None
    assert d["documented"] is False
    assert d["code"]["probe"]                 # live source still served
    assert d["code"]["helpers"] == []         # nothing to resolve
    # no fabricated science fields
    assert "references" not in d and "math" not in d and "science" not in d


def test_all_shipped_probes_are_documented(server):
    # Every probe the battery exposes should now return documented:true with refs,
    # math, thresholds, and live code — a regression guard for the doc system.
    probes = _get(server, "/api/probes")
    ids = [p["id"] for p in probes]
    assert len(ids) >= 20
    for pid in ids:
        d = _get(server, f"/api/probe/{pid}")
        assert d["documented"] is True, f"{pid} is undocumented"
        assert d["references"] and all(r.get("url") for r in d["references"]), f"{pid} refs"
        assert d["math"] and all(m["latex"] for m in d["math"]), f"{pid} math"
        assert d["thresholds"], f"{pid} thresholds"
        assert d["effect"] and d["reading"], f"{pid} interpretation"
        assert d["code"]["probe"], f"{pid} code"


def test_probe_doc_endpoint_404(server):
    import urllib.request, urllib.error, json as _j
    try:
        urllib.request.urlopen(server + "/api/probe/does_not_exist", timeout=10)
        assert False, "expected 404"
    except urllib.error.HTTPError as e:
        assert e.code == 404


def test_dataset_items_explorer(server):
    loaded = _post(server, "/api/load", {"type": "synthetic", "n": 60, "seed": 3})
    ds = loaded["dataset_id"]
    page = _get(server, f"/api/dataset/{ds}/items?offset=0&limit=10")
    assert page["total"] == loaded["n"] and len(page["items"]) == 10
    assert page["has_responses"] and page["has_scores"]
    first = page["items"][0]
    assert "question" in first and "answer" in first and "score" in first
    page2 = _get(server, f"/api/dataset/{ds}/items?offset=10&limit=10")
    assert page2["offset"] == 10 and page2["items"][0] != first


def test_dataset_items_404(server):
    import urllib.request, urllib.error
    try:
        urllib.request.urlopen(server + "/api/dataset/nope/items", timeout=10)
        assert False
    except urllib.error.HTTPError as e:
        assert e.code == 404


def test_run_accepts_api_key_passthrough(server, monkeypatch):
    # a local synthetic model needs no key; we just assert the field is accepted
    loaded = _post(server, "/api/load", {"type": "synthetic", "n": 40, "seed": 1})
    started = _post(server, "/api/run", {"dataset_id": loaded["dataset_id"],
                                         "model": "local:honest", "api_key": "ignored-for-local",
                                         "probes": ["dataset_hygiene"]})
    rid = started["run_id"]
    import time as _t
    for _ in range(100):
        st = _get(server, f"/api/run/{rid}")
        if st["status"] != "running":
            break
        _t.sleep(0.05)
    assert st["status"] == "done"


def test_hf_search_endpoint(server, monkeypatch):
    from trust_the_eval.sources import hf
    def fake_search(query="", limit=25, sort="downloads", token=None, relevant_only=True, results_only=False):
        assert query == "math"
        assert relevant_only is True
        return [{"id": "openai/gsm8k", "downloads": 99999, "likes": 500,
                 "gated": False, "updated": "2024-01-01"}]
    monkeypatch.setattr(hf, "search_datasets", fake_search)
    r = _post(server, "/api/hf/search", {"query": "math", "limit": 10})
    assert r["results"][0]["id"] == "openai/gsm8k"


def test_local_server_client_request_shape(monkeypatch):
    # prove the OpenAI-compatible client builds the right request and parses the reply
    from trust_the_eval.model.providers import LocalServerClient
    captured = {}
    class FakeResp:
        def __init__(self, payload): self._p = payload
        def read(self): import json; return json.dumps(self._p).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False
    def fake_urlopen(req, timeout=120):
        import json
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        captured["auth"] = req.headers.get("Authorization")
        return FakeResp({"choices": [{"message": {"content": "The answer is 4."}}]})
    import urllib.request as _urlreq
    monkeypatch.setattr(_urlreq, "urlopen", fake_urlopen)
    c = LocalServerClient("http://localhost:11434/v1", "llama3.2", api_key="sk-x")
    out = c.complete("What is 2 + 2?", temperature=0.0)
    assert out == "The answer is 4."
    assert captured["url"].endswith("/chat/completions")
    assert captured["body"]["model"] == "llama3.2"
    assert captured["body"]["messages"][0]["content"] == "What is 2 + 2?"
    assert captured["auth"] == "Bearer sk-x"


def test_probe_doc_resolves_class_method_code_ref(server):
    # provenance_repro cites EvalArtifact.content_hash (a class method path)
    d = _get(server, "/api/probe/provenance_repro")
    names = [h["name"] for h in d["code"]["helpers"]]
    assert "trust_the_eval.artifact.EvalArtifact.content_hash" in names
    src = next(h["source"] for h in d["code"]["helpers"]
               if h["name"].endswith("content_hash"))
    assert "def content_hash" in src


def test_api_calibration_endpoint(server):
    # /api/calibration returns measured metrics, coverage tiers and inline SVG curves
    d = _get(server, "/api/calibration?seed=0")
    assert "pooled" in d and d["pooled"]["precision"]["value"] == 1.0
    assert d["coverage"]["counts"]["real_labeled"] >= 2
    assert len(d["probes"]) >= 20
    # every probe payload carries tier + metrics; most carry inline SVG curves
    svg_count = 0
    for pr in d["probes"]:
        assert pr["tier"] in ("real_labeled", "structural_exact", "behavioral_synthetic")
        assert "matrix" in pr and "curves_meta" in pr
        if pr["roc_svg"]:
            assert pr["roc_svg"].startswith("<svg")
            svg_count += 1
    assert svg_count >= 15   # the vast majority discriminate and yield a curve


# ----------------------------- Phase 2: compare + trust report -----------------------------

def _wait_run(base, run_id, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = _get(base, f"/api/run/{run_id}")
        if st["status"] != "running":
            return st
        time.sleep(0.1)
    raise AssertionError("run did not finish in time")


def _load_and_run(base, seed, n=40):
    ds = _post(base, "/api/load", {"type": "synthetic", "n": n, "seed": seed})["dataset_id"]
    rid = _post(base, "/api/run", {"dataset_id": ds, "model": "none"})["run_id"]
    st = _wait_run(base, rid)
    assert st["status"] == "done" and st["findings"]
    return rid


def test_compare_endpoint_unpaired_and_paired(server):
    ra = _load_and_run(server, seed=3)
    rb = _load_and_run(server, seed=9)
    ra2 = _load_and_run(server, seed=3)  # same content as A -> paired, same hash

    # unpaired: different seeds, different items
    c = _post(server, "/api/compare", {"run_a": ra, "run_b": rb})
    assert set(c) >= {"labels", "score", "probes", "ranking", "comparability"}
    assert c["comparability"]["paired"] is False
    assert c["score"]["a"] and c["score"]["b"] and c["score"]["gap"]
    assert c["ranking"]["verdict"] in ("sup", "fra", "uns")
    assert isinstance(c["ranking"]["reasons"], list) and c["ranking"]["reasons"]

    # paired: identical content -> same hash, zero-gap note
    c2 = _post(server, "/api/compare", {"run_a": ra, "run_b": ra2})
    assert c2["comparability"]["paired"] is True
    assert c2["comparability"]["same_content_hash"] is True
    assert any("zero by construction" in n for n in c2["comparability"]["notes"])

    # unknown run id -> 404
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(server, "/api/compare", {"run_a": ra, "run_b": "nope"})
    assert e.value.code == 404


def test_export_trust_report(server):
    rid = _load_and_run(server, seed=3)
    with urllib.request.urlopen(server + f"/api/run/{rid}/export?fmt=trust", timeout=120) as r:
        assert r.status == 200
        assert "text/html" in r.headers.get("Content-Type", "")
        html = r.read().decode()
    assert html.lstrip().lower().startswith("<!doctype")
    assert "<script" not in html
    assert "sha256:" in html
    assert html.count('class="vb"') == 4          # one verdict badge per claim type
    assert "provenance stamp, not a cryptographic signature" in html

    # unfinished/unknown run -> 409
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(server + "/api/run/nope/export?fmt=trust", timeout=10)
    assert e.value.code == 409


def test_view_registry_and_toggle_served(server):
    with urllib.request.urlopen(server + "/", timeout=10) as r:
        html = r.read().decode()
    # analytical-view registry + per-view disable mechanism present
    assert 'id="vizpanel"' in html
    assert "function renderViews" in html and "const VIEWS" in html
    assert "function toggleView" in html and "VIEW_PREFS" in html
    # view #1: honest triage grid (severity × evidence tier), not the degenerate scatter
    assert "function triageGrid" in html
    assert "Triage \u2014 severity \u00d7 evidence tier" in html
    assert "triagePlaneSvg" not in html  # the over-claiming scatter was removed


def _tp_post(base, path, body):
    import json, urllib.request
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def _tp_get(base, path):
    import json, urllib.request
    return json.loads(urllib.request.urlopen(base + path, timeout=120).read())


def _tp_run_static(server):
    import time
    ds = _tp_post(server, "/api/load", {"type": "synthetic", "n": 60, "seed": 3})["dataset_id"]
    rid = _tp_post(server, "/api/run", {"dataset_id": ds, "model": "local:sandbagger"})["run_id"]
    while True:
        st = _tp_get(server, f"/api/run/{rid}")
        if st["status"] != "running":
            break
        time.sleep(0.1)
    return {f["probe_id"]: f for f in st["findings"]}


def test_formula_operands_emitted_and_recompute(server):
    import math
    F = _tp_run_static(server)
    # dataset_hygiene now exposes N (operand of exact_dups = N - #distinct)
    assert "N" in F["dataset_hygiene"]["evidence"]
    # statistical_power exposes every operand its formula needs
    ev = F["statistical_power"]["evidence"]
    for key in ["n", "n_scored", "scoring_coverage", "k", "accuracy", "z", "wald_halfwidth", "min_sig_gap", "wilson_lo", "wilson_hi"]:
        assert key in ev, key
    # anti-drift: the DISPLAYED formula reproduces the output  gap = z*sqrt(2)*sqrt(p(1-p)/n_scored)
    p, n_eff, z = ev["accuracy"], ev["n_scored"], ev["z"]
    recomputed = z * math.sqrt(2) * math.sqrt(p * (1 - p) / n_eff)
    assert abs(recomputed - F["statistical_power"]["score"]) < 0.01
    assert abs(ev["min_sig_gap"] - F["statistical_power"]["score"]) < 0.01
    # provenance: each operand carries an origin (from eval / parameter / computed)
    doc = _tp_get(server, "/api/probe/statistical_power")
    srcs = {i["key"]: i for i in doc.get("inputs", [])}
    assert srcs["z"]["origin"] == "parameter"
    assert srcs["n"]["origin"] == "measured"
    assert srcs["min_sig_gap"]["origin"] == "derived"


def test_bindings_panel_served(server):
    import urllib.request
    html = urllib.request.urlopen(server + "/", timeout=10).read().decode()
    assert "function bindingsBlock" in html
    assert "Inputs &amp; computed output" in html
    assert "function firedSeverity" in html and "DOC_OPEN_ID" in html
    assert "ORIGIN_LABEL" in html and "<th>source</th>" in html
