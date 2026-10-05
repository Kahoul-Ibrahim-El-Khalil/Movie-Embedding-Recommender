#  A Data Science Project: Movie Predictor

This repository contains a data science project built around predicting and analyzing movie-related data sourced from IMDB.

##  Project Structure

- **`data/`**  
  Contains raw `.csv` data files collected from IMDB.

- **`DataCollecting.py`**  
  This module is responsible for fetching and storing movie data from the IMDB API or web scraping.

- **`MoviePredictor.py`**  
  The core logic of the project, including preprocessing, model training, evaluation, and prediction.

## Notebooks

- **`IMDB-surfer.ipynb`**  
  An interactive notebook for exploring and visualizing the movie dataset.

- **`main.ipynb`**  
  The main entry point to run the end-to-end pipeline.

> The project is designed to be executed in a **Jupyter Notebook environment**, allowing for interactive data analysis and model experimentation.

## Command line (recommended)

```bash
python cli.py train    --data 16k --run runs/minilm --epochs 30
python cli.py eval     --run runs/minilm
python cli.py index    --run runs/minilm
python cli.py similar  --run runs/minilm "The Dark Knight" -k 10
python cli.py search   --run runs/minilm "a thief enters dreams to plant an idea"
python cli.py baseline --data small "a thief enters dreams to plant an idea"
python cli.py viz      --run runs/minilm
```

Compare two runs (e.g. frozen vs fine-tuned encoder) side by side:

```bash
python cli.py compare --runs runs/frozen runs/finetuned --query "Snow White" "Barbie" -k 5
```

This prints the config diff, the metric table (baseline / text-only / full per run),
and head-to-head top-k lists with overlap for each query title. Keep data+seed
identical between runs so the test splits match — otherwise it warns you.

`--data` accepts `small` (Data/data.csv, ~409 movies), `16k` (Raw/16k_Movies.csv),
`top` (530 high-rated), `wiki` (Wikipedia plots), or any CSV path. A run directory
(`runs/<name>/`) is the unit of work: training writes dataset, text model, dims,
and splits into `config.json`, and every later command reads them back from there —
no manual `--data`/`--text-model` threading, no stale-cache surprises.

Prefer menus? `python main.py` is a thin wizard over the same commands.

## What this project does exactly

1. **Data** — movie CSVs already in the repo (`DataCollecting.py` output, `Raw/16k_Movies.csv`,
   Wikipedia plots dump) are normalized to one schema: title, year, genres, description,
   director, writer, rating, duration, votes, cast.
2. **Text features** — each movie description is encoded once by a *frozen* pretrained
   sentence-transformer (selectable, default `all-MiniLM-L6-v2`; cached by content hash, so
   retraining or swapping models never reuses stale vectors). No language model is trained
   from scratch.
3. **Learned movie embedding** — a PyTorch model (`recommender/model.py`) fuses text features with
   learned genre multi-hot, cast multi-hot, director/writer embeddings, year, rating, duration,
   and votes projections through an MLP into a single L2-normalized embedding (default 128-D).
   Dimensions are learned, not hand-assigned.
   By default the text encoder is frozen; `--finetune-text` instead fine-tunes MiniLM jointly
   against the same loss (at its own LR, `--text-lr 2e-5`) and saves it to `<run>/text_encoder`.
4. **Training objective** — multi-positive InfoNCE with in-batch negatives (vectorized, logsumexp).
   Positives: movie pairs sharing ≥1 genre (`--jaccard` weights by genre Jaccard instead).
   Every other movie in the batch is a negative. Because genres are both an input and the
   positive-pair definition, training applies **modality dropout** (`--modality-dropout 0.3`):
   each metadata input is randomly zeroed per sample so the model must also learn from text.
5. **Evaluation** — movies are split train/val/test at training time and the *saved* test split
   is reused (never re-split). The trained model is compared against the original pretrained
   MiniLM baseline on Recall@K, Precision@K, HitRate@K, NDCG@K, MRR. The fair column is
   **baseline vs trained text-only** (metadata dropped); the full-input column still sees the
   genres that also define the ground truth. Proxy ground truth is genre overlap — there are
   no user-interaction ratings in this repo, so this is a content-based model, not
   collaborative filtering.
6. **Inference** — the trained model encodes the whole catalog once; a recommendation query is one
   embedding lookup + cosine similarity (matrix multiply), with the query movie excluded.

**Predicting from free text** (`search`, or `predict` in `main.py`):

```bash
python cli.py search --run runs/minilm "a young wizard joins a magic school"
```

The description is run through the text encoder and the trained fusion network in text-only
mode (same `drop=META` the model saw via modality dropout), then cosine-matched against the
cached catalog — no retraining needed.

**Visualizing the embedding space** (`viz`):

```bash
python -m recommender.visualize --out-dir runs/minilm --n 300 --html viz.html
```

Projects the stored catalog embeddings to 3D with PCA and shows an interactive Plotly scatter,
coloured by primary genre (hover for movie titles).

## Version control (`.gitignore`)

Only source, notebooks, and the base `.csv` datasets are meant to be committed. Excluded:
`__pycache__/`, notebook checkpoints, trained weights (`*.pt`, `*.pkl`), and the `artifacts/`
output directories — these are regenerable with `recommender.train` / `recommender.evaluate`.
If the large raw CSVs push the repo size, consider Git LFS or keeping them out of git that way.

## Embeddings

For convenience, embeddings are saved to disk. You can either reuse the stored versions or let the code recompute them when necessary

## Learned PyTorch Recommender (`recommender/`)

The original MiniLM + cosine pipeline (`MoviePredictor.py`) is kept as a baseline.
A movie-specific recommendation model is trained on top of frozen MiniLM text features:

```bash
# 1. Train (16k | small | top | wiki, or a CSV path — schema auto-detected)
python cli.py train --data 16k --run runs/minilm --epochs 30

# with AMD GPU (DirectML) and joint fine-tuning of MiniLM:
python cli.py train --data 16k --run runs/minilm --epochs 30 --device dml --finetune-text

# 2-6. Evaluate, index, query, visualize (all read config from the run dir)
python cli.py eval     --run runs/minilm
python cli.py index    --run runs/minilm
python cli.py similar  --run runs/minilm "The Dark Knight" -k 10
python cli.py search   --run runs/minilm "a young wizard joins a magic school"
python cli.py viz      --run runs/minilm
```

> **AMD GPU notes** (RX 6700 XT, torch 2.13 CPU build currently installed):
> - Easiest on Windows: `pip install torch-directml`, then pass `--device dml`.
> - Faster alternative: run everything inside WSL2 with the Linux ROCm PyTorch build.
> - Rocm is not supported natively on Windows, so CUDA flags will not work on this machine.

# 2. Evaluate baseline vs trained model
python -m recommender.evaluate --out-dir runs/minilm

# 3. Build the catalog index (encode all movies once)
python -m recommender.recommend --out-dir runs/minilm --build-index

# 4. Recommend
python -m recommender.recommend --out-dir runs/minilm --title "The Dark Knight" --k 10
```

Each run directory holds: `config.json`, `model.pt`, `feature_encoder.pkl`,
`text_encoder/` (if fine-tuned), `train/val/test_movies.csv`,
`catalog_embeddings.pt`, `catalog.csv`, `evaluation.json`.

##  Dependencies

See `requirements.txt`: `numpy`, `pandas`, `scikit-learn`, `plotly`, `joblib`, `requests`, `torch`, `sentence-transformers`
