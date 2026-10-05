"""Movie Predictor CLI.

  python cli.py train    --data 16k --run runs/minilm --epochs 30
  python cli.py eval     --run runs/minilm
  python cli.py index    --run runs/minilm
  python cli.py similar  --run runs/minilm "The Dark Knight" -k 10
  python cli.py search   --run runs/minilm "a thief enters dreams to plant an idea"
  python cli.py baseline --data small "a thief enters dreams to plant an idea"
  python cli.py viz      --run runs/minilm
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

DATASETS = {
    "small": "Data/data.csv",                          # DataCollecting output, ~409 movies
    "16k": "Raw/16k_Movies.csv",                       # IMDB dump with ratings/directors
    "top": "Data/random-over-85-from16k.csv",          # 530 high-rated movies
    "wiki": "Data/wiki_movie_plots_deduped.csv",       # Wikipedia plots (slow)
}
TEXT_MODELS = ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L3-v2", "all-mpnet-base-v2", "multi-qa-MiniLM-L6-cos-v1"]


def resolve_data(value: str) -> str:
    return DATASETS.get(value, value)


def load_run(run: str) -> dict:
    cfg = Path(run) / "config.json"
    if not cfg.exists():
        sys.exit(f"No trained run at '{run}'. Train first:\n  python cli.py train --data 16k --run {run}")
    return json.loads(cfg.read_text())


def call(module: str, *args) -> int:
    cmd = [sys.executable, "-m", f"recommender.{module}", *map(str, args)]
    print("$ " + " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


def cmd_train(a):
    args = ["--data", resolve_data(a.data), "--out-dir", a.run, "--epochs", a.epochs,
            "--embed-dim", a.embed_dim, "--text-model", a.text_model, "--device", a.device]
    if a.finetune_text:
        args.append("--finetune-text")
    return call("train", *args)


def cmd_eval(a):
    load_run(a.run)  # config carries data + dims; evaluate reads the saved test split
    return call("evaluate", "--out-dir", a.run, "--device", a.device)


def cmd_index(a):
    load_run(a.run)
    return call("recommend", "--out-dir", a.run, "--build-index", "--device", a.device)


def cmd_similar(a):
    load_run(a.run)
    return call("recommend", "--out-dir", a.run, "--title", a.title, "--k", a.k)


def cmd_search(a):
    load_run(a.run)
    return call("recommend", "--out-dir", a.run, "--description", a.text, "--k", a.k)


def cmd_viz(a):
    load_run(a.run)
    return call("visualize", "--out-dir", a.run)


def cmd_compare(a):
    for r in a.runs:
        load_run(r)
    args = ["--runs", *a.runs, "-k", a.k]
    if a.query:
        args += ["--query", *a.query]
    return call("compare", *args)


def cmd_baseline(a):
    from MoviePredictor import MoviePredictor
    p = MoviePredictor(Raw_Data_Path=resolve_data(a.data), Sentence_Transformer_Model=a.text_model)
    print(p.predict_movie(a.text, Top_K=a.k).drop(columns=["Description"], errors="ignore").to_string())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="movies", description="Movie recommender: train, evaluate, query.")
    sub = p.add_subparsers(dest="command", required=True)

    def run_arg(sp):
        sp.add_argument("--run", default="runs/default", help="run directory (default: runs/default)")

    t = sub.add_parser("train", help="train the recommender and save a run")
    t.add_argument("--data", required=True, help=f"{'|'.join(DATASETS)} or a CSV path")
    run_arg(t)
    t.add_argument("--text-model", default=TEXT_MODELS[0], choices=TEXT_MODELS)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--embed-dim", type=int, default=128)
    t.add_argument("--finetune-text", action="store_true", help="also fine-tune the sentence encoder")
    t.add_argument("--device", default="cpu", help="cpu | cuda | dml")
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("eval", help="compare trained model vs. pretrained baseline")
    run_arg(e); e.add_argument("--device", default="cpu"); e.set_defaults(fn=cmd_eval)

    i = sub.add_parser("index", help="precompute catalog embeddings for fast search")
    run_arg(i); i.add_argument("--device", default="cpu"); i.set_defaults(fn=cmd_index)

    s = sub.add_parser("similar", help="movies similar to a known title")
    run_arg(s); s.add_argument("title"); s.add_argument("-k", type=int, default=10); s.set_defaults(fn=cmd_similar)

    q = sub.add_parser("search", help="movies matching a free-text plot description")
    run_arg(q); q.add_argument("text"); q.add_argument("-k", type=int, default=10); q.set_defaults(fn=cmd_search)

    v = sub.add_parser("viz", help="3D PCA plot of the embedding space")
    run_arg(v); v.set_defaults(fn=cmd_viz)

    c = sub.add_parser("compare", help="side-by-side metrics + recommendations for 2+ runs")
    c.add_argument("--runs", nargs="+", required=True, help="run directories to compare")
    c.add_argument("--query", nargs="*", default=[], help="title queries for head-to-head recommendations")
    c.add_argument("-k", type=int, default=5); c.set_defaults(fn=cmd_compare)

    b = sub.add_parser("baseline", help="no training: plain sentence-embedding search")
    b.add_argument("text"); b.add_argument("--data", default="small")
    b.add_argument("--text-model", default=TEXT_MODELS[0], choices=TEXT_MODELS)
    b.add_argument("-k", type=int, default=5); b.set_defaults(fn=cmd_baseline)
    return p


def main() -> None:
    args = build_parser().parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()
