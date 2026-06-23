from __future__ import annotations
import argparse
import json
import sys
from typing import Optional

from . import probes as _probes  # noqa: F401  (self-registers all 20 probes)
from .adapters import generic_json, inspect_log, promptfoo
from .emit import html as html_mod, otel, report as report_mod
from .model import build_model
from .runner import run_battery


def _load_artifact(path: str):
    if path.endswith(".eval"):
        return inspect_log.load(path)
    if "promptfoo" in path:
        return promptfoo.load(path)
    return generic_json.load(path)


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        prog="trust-the-eval",
        description="Red-team an evaluation RESULT for validity (not the model).")
    p.add_argument("command", choices=["check", "list-probes", "serve", "calibrate"])
    p.add_argument("artifact", nargs="?", help="path to an eval result (json / .eval / promptfoo)")
    p.add_argument("--model", default=None,
                   help="local:<name> | api:<base_url>#<model> | anthropic:<model> | openai:<model>")
    p.add_argument("--probes", default=None, help="comma-separated probe ids to run")
    p.add_argument("--otel-out", default=None)
    p.add_argument("--html-out", default=None)
    p.add_argument("--trust-report", default=None,
                   help="write a self-contained, provenance-stamped Run Trust Report (HTML)")
    p.add_argument("--json-out", default=None)
    p.add_argument("--port", type=int, default=8077, help="port for 'serve'")
    p.add_argument("--host", default="127.0.0.1", help="host for 'serve'")
    p.add_argument("--no-browser", action="store_true", help="don't auto-open the browser")
    p.add_argument("--seed", type=int, default=0, help="seed for 'calibrate'")
    p.add_argument("--mmlu-redux", default=None,
                   help="path to a local MMLU-Redux json/jsonl export "
                        "(adds the real-world acid test to 'calibrate')")
    p.add_argument("--report-html", default=None,
                   help="write the published calibration report (self-contained HTML)")
    args = p.parse_args(argv)

    if args.command == "calibrate":
        from .calibration import (run_calibration, summary_table,
                                  cases_from_mmlu_redux, load_mmlu_redux_rows,
                                  render_report_html)
        extra = {}
        if args.mmlu_redux:
            rows = load_mmlu_redux_rows(path=args.mmlu_redux)
            from .model.local import HonestModel
            cases = cases_from_mmlu_redux(rows, seed=args.seed)
            for c in cases:
                c.model = HonestModel(seed=args.seed)
            extra["label_error_audit"] = cases
            print(f"loaded {len(rows)} MMLU-Redux rows -> {len(cases)} real cases\n")
        rep = run_calibration(seed=args.seed, extra_cases=extra)
        print(summary_table(rep))
        if args.json_out:
            json.dump(rep.to_dict(), open(args.json_out, "w"), indent=2, ensure_ascii=False)
            print(f"\ncalibration json -> {args.json_out}")
        if args.report_html:
            from .probe import get_probe

            def _caveat(pid):
                try:
                    c = (get_probe(pid)().DOC.caveats or "").strip()
                    first = c.split(". ")[0]
                    return (first[:240] + ("\u2026" if len(first) > 240 else "")) if first else None
                except Exception:
                    return None
            caveats = {c.probe_id: _caveat(c.probe_id) for c in rep.probes}
            open(args.report_html, "w", encoding="utf-8").write(
                render_report_html(rep, caveats=caveats))
            print(f"published report -> {args.report_html}")
        # exit non-zero if any probe fails to discriminate (recall or specificity < 0.5)
        weak = [c.probe_id for c in rep.probes
                if (c.matrix.recall().value or 0) < 0.5
                or (c.matrix.specificity().value or 0) < 0.5]
        if weak:
            print(f"\nUNDISCRIMINATING PROBES: {', '.join(weak)}", file=sys.stderr)
            return 1
        return 0

    if args.command == "serve":
        from .ui.server import serve
        return serve(host=args.host, port=args.port, open_browser=not args.no_browser)


    if args.command == "list-probes":
        from .probe import all_probes
        for c in sorted(all_probes(), key=lambda c: c.id):
            tag = "model" if c().requires_model else "static"
            print(f"{c.id:28} [{tag:6}] {c.paper_priority:14} {c.name}")
        return 0

    if not args.artifact:
        p.error("artifact path required for 'check'")
    artifact = _load_artifact(args.artifact)
    model = build_model(args.model)
    probe_ids = args.probes.split(",") if args.probes else None
    rep = run_battery(artifact, model=model, probe_ids=probe_ids)

    print(report_mod.to_console(rep))
    if args.otel_out:
        json.dump(otel.to_otel_event(rep), open(args.otel_out, "w"), indent=2, ensure_ascii=False)
        print(f"\nemitted: gen_ai.eval.trust -> {args.otel_out}")
    if args.json_out:
        json.dump(report_mod.to_dict(rep), open(args.json_out, "w"), indent=2, ensure_ascii=False)
        print(f"report json -> {args.json_out}")
    if args.html_out:
        open(args.html_out, "w").write(html_mod.to_html(rep))
        print(f"html report -> {args.html_out}")
    if args.trust_report:
        from .emit.trust_report import render_trust_report
        open(args.trust_report, "w", encoding="utf-8").write(
            render_trust_report(rep, artifact, model_spec=args.model or "none"))
        print(f"trust report (provenance-stamped, self-contained) -> {args.trust_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
