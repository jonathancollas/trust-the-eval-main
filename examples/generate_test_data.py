"""Generate a synthetic eval-result with known, baked-in validity defects.

Thin CLI wrapper around `trust_the_eval.datagen` (the single source of truth).

Usage:
    python examples/generate_test_data.py [out.json] [--n N] [--seed S]
"""
from __future__ import annotations
import argparse
import json
import os

from trust_the_eval import datagen


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default=os.path.join(here, "generated_eval.json"))
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    data = datagen.generate(n=args.n, seed=args.seed)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

    items = data["items"]
    n_mcq = sum(1 for it in items if "options" in it)
    cats = {}
    for it in items:
        cats[it["category"]] = cats.get(it["category"], 0) + 1
    print(f"wrote {len(items)} items ({n_mcq} MCQ) -> {args.out}")
    print("categories:", dict(sorted(cats.items())))


if __name__ == "__main__":
    main()
