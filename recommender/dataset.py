"""Data loading and feature encoding for the movie recommender.

Supports the two dataset layouts present in this repository:

* DataCollecting output (e.g. ``Data/data.csv``):
      Title, Year, Genre, Description
* IMDB raw dump (e.g. ``Raw/16k_Movies.csv``):
      Title, Release Date, Description, Rating, No of Persons Voted,
      Directed by, Written by, Duration, Genres

Both are normalized into a single schema:

    title, year, genres (list[str]), description, director, rating
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

UNIFIED_COLUMNS = ["title", "year", "genres", "description", "director", "rating"]

MAX_DIRECTORS = 2000  # cap director vocabulary; rest map to <UNK>


def _parse_year_from_date(value) -> float:
    """Extract the year from strings like 'Mar 22, 1996' or '1996'."""
    if pd.isna(value):
        return np.nan
    s = str(value)
    digits = "".join(ch if ch.isdigit() else " " for ch in s).split()
    for tok in digits:
        if len(tok) == 4 and 1800 <= int(tok) <= 2100:
            return float(tok)
    return np.nan


def _split_genres(value) -> List[str]:
    if pd.isna(value):
        return []
    return [g.strip() for g in str(value).split(",") if g.strip() and g.strip().upper() != "N/A"]


def _pick(cols: dict, *names):
    for n in names:
        if n in cols:
            return cols[n]
    return None


def load_movies(csv_path: str | Path) -> pd.DataFrame:
    """Load a movie CSV and normalize it to the unified schema."""
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    nan_series = lambda: pd.Series(np.nan, index=df.index)

    desc_col = _pick(cols, "description", "plot")
    title_col = _pick(cols, "title")
    genre_col = _pick(cols, "genre", "genres")
    for label, c in (("description/plot", desc_col), ("title", title_col), ("genre", genre_col)):
        if c is None:
            raise ValueError(f"No {label} column found in {csv_path}")

    def txt(*names):
        c = _pick(cols, *names)
        return df[c].fillna("<missing>").astype(str) if c else pd.Series("<missing>", index=df.index)

    def num(*names):
        c = _pick(cols, *names)
        return pd.to_numeric(df[c], errors="coerce") if c else nan_series()

    year_col = _pick(cols, "year", "release date", "release year")
    year = df[year_col].apply(_parse_year_from_date) if year_col else nan_series()

    dur_col = _pick(cols, "duration")
    duration = (
        pd.to_numeric(df[dur_col].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
        if dur_col else nan_series()
    )

    cast_col = _pick(cols, "cast")
    cast = (
        df[cast_col].apply(_split_genres)
        if cast_col else pd.Series([[] for _ in range(len(df))], index=df.index)
    )

    out = pd.DataFrame(
        {
            "title": df[title_col],
            "year": year,
            "genres": df[genre_col].apply(_split_genres),
            "description": df[desc_col].fillna("").astype(str).str.strip(),
            "director": txt("directed by", "director"),
            "rating": num("rating"),
            "writer": txt("written by", "writer"),
            "duration": duration,
            "votes": num("no of persons voted", "votes"),
            "cast": cast,
        }
    )
    out = out.dropna(subset=["title"])                      # drop NaN BEFORE casting to str
    out["title"] = out["title"].astype(str).str.strip()
    out = out[(out["description"].str.len() > 0) & (out["description"].str.upper() != "N/A")]
    out = out.drop_duplicates(subset=["title", "year"]).reset_index(drop=True)
    return out


def load_split(out_dir: str | Path, name: str) -> pd.DataFrame:
    """Load train/val/test split exactly as train.py saved it."""
    df = pd.read_csv(Path(out_dir) / f"{name}_movies.csv")
    for c in ("genres", "cast"):
        df[c] = df[c].apply(_split_genres)
    for c in ("director", "writer"):
        df[c] = df[c].fillna("<missing>").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    return df


@dataclass
class FeatureEncoder:
    """Fit-on-train-only encoder that converts raw fields to model inputs.

    Produces, per movie:
      * genre multi-hot vector  (float32, shape [n_genres])
      * normalized year         (float32 scalar)
      * normalized rating       (float32 scalar)
      * director index          (int64 scalar, 0 reserved for <unk>/<missing>)
    """

    genre_vocab: List[str] = field(default_factory=list)
    director_vocab: List[str] = field(default_factory=list)
    writer_vocab: List[str] = field(default_factory=list)
    cast_vocab: List[str] = field(default_factory=list)
    year_min: float = 1900.0
    year_max: float = 2025.0
    rating_mean: float = 6.5
    rating_std: float = 1.5
    duration_mean: float = 100.0
    duration_std: float = 40.0
    votes_log_mean: float = 10.0
    votes_log_std: float = 2.0

    @property
    def n_genres(self) -> int:
        return len(self.genre_vocab)

    @property
    def n_directors(self) -> int:
        return len(self.director_vocab) + 1  # +1 for <unk> at index 0

    @property
    def n_writers(self) -> int:
        return len(self.writer_vocab) + 1

    @property
    def n_cast(self) -> int:
        return len(self.cast_vocab)

    def fit(self, df: pd.DataFrame) -> "FeatureEncoder":
        genre_set = set()
        for gs in df["genres"]:
            genre_set.update(gs)
        self.genre_vocab = sorted(genre_set)

        self.director_vocab = [d for d in df["director"].value_counts().index if d != "<missing>"][:MAX_DIRECTORS]
        self.writer_vocab = [w for w in df["writer"].value_counts().index if w != "<missing>"][:MAX_DIRECTORS]

        cast_counts = pd.Series([c for casts in df["cast"] for c in casts]).value_counts()
        self.cast_vocab = cast_counts.index[:MAX_DIRECTORS].tolist()

        years = df["year"].dropna()
        if len(years):
            self.year_min, self.year_max = float(years.min()), float(years.max())
        ratings = df["rating"].dropna()
        if len(ratings):
            self.rating_mean, self.rating_std = float(ratings.mean()), float(ratings.std() or 1.5)
        durations = df["duration"].dropna()
        if len(durations):
            self.duration_mean, self.duration_std = float(durations.mean()), float(durations.std() or 40.0)
        votes = df["votes"].dropna()
        if len(votes):
            logv = np.log1p(votes)
            self.votes_log_mean, self.votes_log_std = float(logv.mean()), float(logv.std() or 2.0)
        return self

    def encode(self, df: pd.DataFrame) -> dict:
        import torch

        genre_idx = {g: i for i, g in enumerate(self.genre_vocab)}
        genres = np.zeros((len(df), len(self.genre_vocab)), dtype=np.float32)
        for i, gs in enumerate(df["genres"]):
            for g in gs:
                if g in genre_idx:
                    genres[i, genre_idx[g]] = 1.0

        cast_idx = {c: i for i, c in enumerate(self.cast_vocab)}
        cast = np.zeros((len(df), max(len(self.cast_vocab), 1)), dtype=np.float32)
        for i, cs in enumerate(df["cast"]):
            for c in cs:
                if c in cast_idx:
                    cast[i, cast_idx[c]] = 1.0

        span = max(self.year_max - self.year_min, 1.0)
        year = df["year"].fillna((self.year_min + self.year_max) / 2)
        year_norm = ((year - self.year_min) / span).clip(0, 1).to_numpy(dtype=np.float32)

        rating = df["rating"].fillna(self.rating_mean)
        rating_norm = ((rating - self.rating_mean) / max(self.rating_std, 1e-6)).to_numpy(dtype=np.float32)

        duration = df["duration"].fillna(self.duration_mean)
        duration_norm = ((duration - self.duration_mean) / max(self.duration_std, 1e-6)).to_numpy(dtype=np.float32)

        votes = df["votes"].fillna(0)
        votes_norm = ((np.log1p(votes) - self.votes_log_mean) / max(self.votes_log_std, 1e-6)).to_numpy(dtype=np.float32)

        dir_idx = {d: i + 1 for i, d in enumerate(self.director_vocab)}
        director = df["director"].map(lambda d: dir_idx.get(d, 0)).to_numpy(dtype=np.int64)

        wr_idx = {w: i + 1 for i, w in enumerate(self.writer_vocab)}
        writer = df["writer"].map(lambda w: wr_idx.get(w, 0)).to_numpy(dtype=np.int64)

        return {
            "genres": torch.from_numpy(genres),
            "cast": torch.from_numpy(cast),
            "year": torch.from_numpy(year_norm).unsqueeze(1),
            "rating": torch.from_numpy(rating_norm).unsqueeze(1),
            "duration": torch.from_numpy(duration_norm).unsqueeze(1),
            "votes": torch.from_numpy(votes_norm).unsqueeze(1),
            "director": torch.from_numpy(director),
            "writer": torch.from_numpy(writer),
        }

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureEncoder":
        with open(path, "rb") as f:
            state = pickle.load(f)
        enc = cls()
        enc.__dict__.update(state)
        return enc


def genre_overlap(a: List[str], b: List[str]) -> bool:
    return len(set(a) & set(b)) > 0


def positive_mask(genres: List[List[str]]) -> np.ndarray:
    """Boolean matrix: mask[i, j] True when movies i, j share a genre (vectorized)."""
    vocab = {g: i for i, g in enumerate(sorted({g for gs in genres for g in gs}))}
    m = np.zeros((len(genres), max(len(vocab), 1)), dtype=np.float32)
    for i, gs in enumerate(genres):
        for g in gs:
            m[i, vocab[g]] = 1.0
    mask = (m @ m.T) > 0
    np.fill_diagonal(mask, False)
    return mask


def train_val_test_split(
    df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42
):
    idx = np.random.default_rng(seed).permutation(len(df))
    n_test = int(len(df) * test_frac)
    n_val = int(len(df) * val_frac)
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )
