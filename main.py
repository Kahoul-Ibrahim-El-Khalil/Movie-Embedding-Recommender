"""Interactive entry point for the Movie Predictor project.

Run:
    python main.py

Guides you through:
  1. choosing a dataset
  2. choosing the similarity backend:
       - 'baseline'  : generic SentenceTransformer embeddings + sklearn cosine (original approach)
       - 'pytorch'   : trained MovieRecommender embeddings + cosine (new approach)
  3. choosing the sentence-embedding model
  4. training / evaluating / indexing / recommending
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.resolve()

DATASETS = {
    "1": ("Data/data.csv", "DataCollecting output, 409 movies (Title, Year, Genre, Description)"),
    "2": ("Raw/16k_Movies.csv", "16k IMDB movies with Rating, Director, Genres"),
    "3": ("Data/random-over-85-from16k.csv", "530 high-rated movies"),
    "4": ("Data/wiki_movie_plots_deduped.csv", "Wikipedia plots, Cast, Director (slow)"),
}

TEXT_MODELS = {
    "1": "all-MiniLM-L6-v2",
    "2": "paraphrase-MiniLM-L3-v2",
    "3": "all-mpnet-base-v2",
    "4": "multi-qa-MiniLM-L6-cos-v1",
}


def ask(prompt: str, options: dict, default: str | None = None) -> str:
    print(f"\n{prompt}")
    for key, val in options.items():
        label = val[0] if isinstance(val, tuple) else val
        desc = val[1] if isinstance(val, tuple) else ""
        print(f"  [{key}] {label}  {desc}")
    choice = input("  > ").strip()
    if not choice:
        if default is None:
            print("  Please choose one.")
            return ask(prompt, options, default)
        choice = default  # resolve the default key through the mapping below
    if choice in options:
        return options[choice][0] if isinstance(options[choice], tuple) else options[choice]
    print("  Invalid choice, try again.")
    return ask(prompt, options, default)


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_DIR, check=False)


def main() -> None:
    print("=" * 60)
    print(" MOVIE PREDICTOR — interactive entry point")
    print("=" * 60)

    data = ask("Select a dataset:", DATASETS)

    backend = ask(
        "Select similarity backend:",
        {
            "1": ("baseline", "Original approach: frozen SentenceTransformer + sklearn cosine similarity"),
            "2": ("pytorch", "New approach: trained PyTorch MovieRecommender embeddings + cosine"),
        },
        default="2",
    )

    text_model = ask("Select the sentence-embedding model:", TEXT_MODELS, default="1")

    action = ask(
        "Select an action:",
        {
            "1": ("train", "Train or fine-tune the recommender"),
            "2": ("evaluate", "Compare baseline vs trained model metrics"),
            "3": ("build-index", "Precompute catalog embeddings for fast search"),
            "4": ("recommend", "Movies similar to a known title"),
            "5": ("predict", "Matches for a free-text description"),
            "6": ("viz", "Interactive 3D PCA plot of the embedding space"),
            "7": ("baseline-only", "Original MoviePredictor pipeline (no training)"),
        },
    )

    out_dir = input("\nArtifacts directory [artifacts]: ").strip() or "artifacts"

    if action == "train":
        if backend == "baseline":
            print("The baseline has nothing to train — it uses a frozen SentenceTransformer.")
            return
        epochs = input("Epochs [30]: ").strip() or "30"
        embed_dim = input("Embedding dimension [128]: ").strip() or "128"
        ft = input("Fine-tune MiniLM too? [y/N]: ").strip().lower() == "y"
        dev = input("Device [cpu/cuda/dml]: ").strip() or "cpu"
        cmd = [sys.executable, "-m", "recommender.train", "--data", data, "--epochs", epochs,
               "--embed-dim", embed_dim, "--text-model", text_model, "--out-dir", out_dir,
               "--device", dev]
        if ft:
            cmd.append("--finetune-text")
        run(cmd)

    elif action == "evaluate":
        # dataset + text model come from the run directory (config.json)
        run([sys.executable, "-m", "recommender.evaluate", "--out-dir", out_dir])

    elif action == "build-index":
        run([sys.executable, "-m", "recommender.recommend", "--out-dir", out_dir, "--build-index"])

    elif action == "recommend":
        title = input("Movie title: ").strip()
        k = input("Top-k [10]: ").strip() or "10"
        if backend == "baseline":
            print("\nBaseline mode: use MoviePredictor.predict_movie(description) with a plot description.")
            return
        run([sys.executable, "-m", "recommender.recommend", "--out-dir", out_dir,
             "--title", title, "--k", k])

    elif action == "predict":
        description = input("Enter a plot description: ").strip()
        k = input("Top-k [10]: ").strip() or "10"
        if backend == "baseline":
            code = (
                "import MoviePredictor\n"
                "p = MoviePredictor.MoviePredictor()\n"
                "print(p.predict_movie(" + repr(description) + ", Top_K=int(" + k + ")).to_string())\n"
            )
            run([sys.executable, "-c", code])
        else:
            run([sys.executable, "-m", "recommender.recommend", "--out-dir", out_dir,
                 "--description", description, "--k", k])

    elif action == "viz":
        run([sys.executable, "-m", "recommender.visualize", "--out-dir", out_dir])

    elif action == "baseline-only":
        description = input("Enter a plot description (free text): ").strip()
        code = (
            "import MoviePredictor\n"
            "p = MoviePredictor.MoviePredictor()\n"
            "print(p.predict_movie(" + repr(description) + ", Top_K=5).to_string())\n"
        )
        run([sys.executable, "-c", code])


if __name__ == "__main__":
    main()
