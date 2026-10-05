"""Recommendation service: encode the catalog once, serve fast cosine queries.

Usage:
    # one-time catalog indexing after training
    python -m recommender.recommend --out-dir runs/minilm index

    # queries (read dataset + dims from the run directory)
    python -m recommender.recommend --out-dir runs/minilm --title "The Dark Knight" --k 10
    python -m recommender.recommend --out-dir runs/minilm --description "a thief enters dreams" --k 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .dataset import FeatureEncoder, _split_genres, load_movies
from .features import compute_text_features, get_device
from .model import META, MovieRecommender


class Recommender:
    def __init__(self, out_dir: str | Path = "artifacts", device: str = "cpu"):
        out_dir = Path(out_dir)
        self.out_dir = out_dir
        self.device = get_device(device)
        self.config = json.loads((out_dir / "config.json").read_text())
        self.encoder = FeatureEncoder.load(out_dir / "feature_encoder.pkl")
        self.model = MovieRecommender(
            text_dim=self.config["text_dim"],
            n_genres=self.config["n_genres"],
            n_directors=self.config["n_directors"],
            n_writers=self.config.get("n_writers", 100),
            n_cast=self.config.get("n_cast", 0),
            use_cast=self.config.get("use_cast", False),
            embed_dim=self.config["embed_dim"],
            hidden_dim=self.config["hidden_dim"],
            dropout=self.config["dropout"],
        ).to(self.device)
        try:
            self.model.load_state_dict(torch.load(out_dir / "model.pt", map_location=self.device, weights_only=True))
        except RuntimeError as e:
            raise RuntimeError(
                f"{e}\n\nThe saved model in {out_dir}/ was trained with an older architecture. "
                f"Re-train it: python -m recommender.train --data <csv> --out-dir {out_dir}"
            ) from e
        self.model.eval()
        self.catalog_embs: torch.Tensor | None = None
        self.catalog_df: pd.DataFrame | None = None
        self._st_cache: dict = {}

    def build_index(self, df: pd.DataFrame, text_feats: torch.Tensor, device=None) -> None:
        device = device or self.device
        enc = self.encoder.encode(df)
        with torch.no_grad():
            self.catalog_embs = self.model(
                text_feats.to(device), {k: v.to(device) for k, v in enc.items()}
            ).cpu()
        self.catalog_df = df.reset_index(drop=True)
        torch.save(self.catalog_embs, self.out_dir / "catalog_embeddings.pt")
        self.catalog_df.assign(
            genres=self.catalog_df["genres"].apply(",".join),
            cast=self.catalog_df["cast"].apply(",".join),
        ).to_csv(self.out_dir / "catalog.csv", index=False)

    def load_index(self) -> None:
        self.catalog_embs = torch.load(self.out_dir / "catalog_embeddings.pt", weights_only=True)
        df = pd.read_csv(self.out_dir / "catalog.csv")
        df["genres"] = df["genres"].apply(_split_genres)
        df["cast"] = df["cast"].apply(_split_genres) if "cast" in df.columns else [[] for _ in range(len(df))]
        for c in ("director", "writer"):
            if c in df.columns:
                df[c] = df[c].fillna("<missing>").astype(str)
        self.catalog_df = df

    def _check_index(self) -> None:
        if self.catalog_embs.shape[1] != self.config["embed_dim"]:
            raise RuntimeError(
                f"catalog_embeddings.pt has dim {self.catalog_embs.shape[1]} but config.json/model.pt "
                f"use dim {self.config['embed_dim']}. The catalog is stale — rebuild it:\n"
                f"  python -m recommender.recommend --out-dir {self.out_dir} --build-index"
            )

    def _text_encoder(self, text_model: str | None = None):
        tm = text_model or self.config.get("text_model", "all-MiniLM-L6-v2")
        if (self.out_dir / "text_encoder").exists():
            tm = str(self.out_dir / "text_encoder")
        if tm not in self._st_cache:
            from sentence_transformers import SentenceTransformer
            self._st_cache[tm] = SentenceTransformer(tm)
        return self._st_cache[tm]

    def find_movie(self, title: str) -> int:
        titles = self.catalog_df["title"].str.lower()
        exact = titles[titles == title.lower()]
        if len(exact):
            return int(exact.index[0])
        partial = titles[titles.str.contains(title.lower(), regex=False)]
        if len(partial):
            return int(partial.index[0])
        raise KeyError(f"Unknown movie title: {title!r}")

    def recommend(self, title: str, k: int = 10) -> pd.DataFrame:
        if self.catalog_embs is None:
            self.load_index()
        self._check_index()
        idx = self.find_movie(title)
        sims = self.catalog_embs @ self.catalog_embs[idx]
        sims[idx] = -1.0  # exclude the query movie itself
        top = torch.argsort(sims, descending=True)[:k]
        result = self.catalog_df.iloc[top.numpy()].copy()
        result["score"] = sims[top].numpy()
        return result[["title", "year", "genres", "rating", "score"]]

    @torch.no_grad()
    def predict_description(self, description: str, k: int = 10, text_model: str | None = None) -> pd.DataFrame:
        """Predict likely movies from a free-text plot description.

        Uses the trained model in text-only mode (metadata dropped), matching
        how it was regularized during training via modality dropout.
        """
        if self.catalog_embs is None:
            self.load_index()
        self._check_index()
        st = self._text_encoder(text_model)
        text = torch.from_numpy(np.asarray(st.encode([description]), dtype=np.float32)).to(self.device)

        zeros = torch.zeros(1, self.encoder.n_genres)
        cast_dim = max(self.encoder.n_cast, 1)
        features = {
            "genres": zeros,
            "cast": torch.zeros(1, cast_dim),
            "director": torch.zeros(1, dtype=torch.long),
            "writer": torch.zeros(1, dtype=torch.long),
            "year": torch.full((1, 1), 0.5),
            "rating": torch.zeros(1, 1),
            "duration": torch.zeros(1, 1),
            "votes": torch.zeros(1, 1),
        }
        features = {k: v.to(self.device) for k, v in features.items()}
        emb = self.model(text, features, drop=META).cpu()[0]

        sims = self.catalog_embs @ emb
        top = torch.argsort(sims, descending=True)[:k]
        result = self.catalog_df.iloc[top.numpy()].copy()
        result["score"] = sims[top].numpy()
        return result[["title", "year", "genres", "rating", "score"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None, help="CSV path; defaults to the run's training data")
    p.add_argument("--out-dir", default="artifacts")
    p.add_argument("--title", default=None)
    p.add_argument("--description", default=None, help="Free-text plot description for prediction")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--build-index", action="store_true")
    p.add_argument("--text-model", default=None, help="Defaults to the model used at training time")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "dml"])
    args = p.parse_args()

    rec = Recommender(out_dir=args.out_dir, device=args.device)
    if args.build_index:
        data = args.data or rec.config.get("data")
        if not data:
            raise SystemExit("No dataset: pass --data <csv> or train with --data first.")
        df = load_movies(data)
        tm = args.text_model or rec.config.get("text_model", "all-MiniLM-L6-v2")
        if (Path(args.out_dir) / "text_encoder").exists():
            tm = str(Path(args.out_dir) / "text_encoder")
        feats = compute_text_features(
            df,
            model_name=tm,
            cache_path=Path(args.out_dir) / "text_feats_catalog.pt",
        )
        rec.build_index(df, feats, device=rec.device)
        print(f"Indexed {len(df)} movies -> {args.out_dir}/catalog_embeddings.pt")

    if args.title:
        print(rec.recommend(args.title, k=args.k).to_string(index=False))
    if args.description:
        print(rec.predict_description(args.description, k=args.k, text_model=args.text_model).to_string(index=False))


if __name__ == "__main__":
    main()
