from __future__ import annotations
import json
import os
import sys

from trust_the_eval.adapters import generic_json
from trust_the_eval.emit import otel, report as report_mod
from trust_the_eval.model.local import HonestModel
from trust_the_eval.runner import run_battery
from trust_the_eval import probes as _probes  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "sample_eval_result.json")
    art = generic_json.load(path)
    print("### static-only battery (no model)\n")
    print(report_mod.to_console(run_battery(art, model=None)))
    print("\n\n### full battery against a HONEST local model\n")
    rep = run_battery(art, model=HonestModel())
    print(report_mod.to_console(rep))
    print("\n--- gen_ai.eval.trust event ---")
    print(json.dumps(otel.to_otel_event(rep), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
