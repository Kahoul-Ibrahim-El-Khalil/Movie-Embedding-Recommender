"""Compare two or more trained runs side by side.

Quantitative: metrics from each run's evaluation.json (baseline MiniLM,
trained text-only = fair column, trained full) in one table.
Qualitative: same title queries against each run's catalog index, with
top-k overlap (Jaccard) between runs.

Usage:
    python -m recommender.compare --runs runs/frozen runs/finetuned --query "Snow White" -k 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_eval(run: str) -> tuple[dict, dict]:
    run_dir = Path(run)
    cfg = json.loads((run_dir / "config.json").read_text())
    ev_path = run_dir / "evaluation.json"
    if not ev_path.exists():
        raise SystemExit(f"No evaluation.json in '{run}'. Run first:\n  python cli.py eval --run {run}")
    return cfg, json.loads(ev_path.read_text())


def print_configs(configs: list[dict], runs: list[str]) -> None:
    keys = ["data", "text_model", "finetune_text", "embed_dim", "hidden_dim",
            "modality_dropout", "jaccard", "epochs", "seed"]
    print(f"\n{'setting':<18}" + "".join(f"{r:>22}" for r in runs))
    for k in keys:
        print(f"{k:<18}" + "".join(f"{str(c.get(k, '-')):>22}" for c in configs))
    same_test = len({(c.get("data"), c.get("seed")) for c in configs}) == 1
    if not same_test:
        print("\nWARNING: runs use different data/seed -> test splits differ, "
              "metrics are not strictly comparable.")


def print_metrics(evals: list[dict], runs: list[str]) -> None:
    variants = [("baseline MiniLM", "baseline"), ("trained text-only", "trained_text_only"), ("trained full", "trained")]
    metrics = sorted(evals[0]["baseline"].keys())
    for label, vkey in variants:
        print(f"\n--- {label} ---")
        print(f"{'metric':<14}" + "".join(f"{r:>22}" for r in runs))
        for m in metrics:
            print(f"{m:<14}" + "".join(f"{ev[vkey][m]:>22.4f}" for ev in evals))


def qualitative(runs: list[str], queries: list[str], k: int) -> None:
    from .recommend import Recommender

    recs = []
    for r in runs:
        try:
            recs.append(Recommender(out_dir=r))
        except Exception as e:
            print(f"\nSkipping qualitative for '{r}': {e}\n  (run: python cli.py index --run {r})")
            recs.append(None)
    if all(r is None for r in recs):
        return

    for q in queries:
        print(f"\n=== similar to {q!r} (top-{k}) ===")
        tops = []
        for r, rec in zip(runs, recs):
            if rec is None:
                tops.append(None)
                continue
            try:
                df = rec.recommend(q, k=k)
                tops.append(df["title"].tolist())
                print(f"  [{r}]")
                for t, s in zip(df["title"], df["score"]):
                    print(f"    {t}  ({s:.3f})")
            except KeyError:
                print(f"  [{r}] title not in catalog, skipped")
                tops.append(None)
        valid = [(r, t) for r, t in zip(runs, tops) if t is not None]
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                a, b = set(valid[i][1]), set(valid[j][1])
                jac = len(a & b) / max(len(a | b), 1)
                print(f"  overlap {valid[i][0]} vs {valid[j][0]}: "
                      f"{len(a & b)}/{k} shared (Jaccard {jac:.2f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="run directories to compare")
    p.add_argument("--query", nargs="*", default=[], help="title queries for head-to-head recommendations")
    p.add_argument("-k", type=int, default=5)
    args = p.parse_args()

    configs, evals = zip(*[load_eval(r) for r in args.runs])
    print_configs(list(configs), args.runs)
    print_metrics(list(evals), args.runs)
    if args.query:
        qualitative(args.runs, args.query, args.k)


if __name__ == "__main__":
    main()
