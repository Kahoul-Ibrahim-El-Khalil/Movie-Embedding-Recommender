"""PyTorch movie recommendation model.

Each input modality is projected into a common `embed_dim` space, concatenated,
and fused by an MLP into the final L2-normalized embedding:

    Description ──► frozen MiniLM (precomputed) ──► Linear ─┐
    Genres (multi-hot) ──────────────────────────► Linear ──┤
    Cast (multi-hot) ─────────────────────────────► Linear ──┤
    Director (id) ──► nn.Embedding ──► Linear ──────────────┤
    Writer (id) ──► nn.Embedding ──► Linear ────────────────┤──► MLP ──► L2-norm ──► z
    Year (scalar) ──► Linear ───────────────────────────────┤
    Rating (scalar) ──► Linear ─────────────────────────────┤
    Duration (scalar) ──► Linear ───────────────────────────┤
    Votes (scalar) ──► Linear ──────────────────────────────┘

`use_cast`/`use_writer` let a CSV without those columns drop the corresponding
inputs entirely (the fusion layer adapts its input size).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MovieRecommender(nn.Module):
    META = ("genres", "director", "writer", "year", "rating", "duration", "votes", "cast")

    def __init__(
        self,
        text_dim: int = 384,
        n_genres: int = 20,
        n_directors: int = 100,
        n_writers: int = 100,
        n_cast: int = 0,
        use_cast: bool = False,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        director_emb_dim: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_cast = use_cast and n_cast > 0

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, embed_dim)
        )
        self.genre_proj = nn.Linear(n_genres, embed_dim)
        self.director_emb = nn.Embedding(n_directors, director_emb_dim)
        self.director_proj = nn.Linear(director_emb_dim, embed_dim)
        self.writer_emb = nn.Embedding(n_writers, director_emb_dim)
        self.writer_proj = nn.Linear(director_emb_dim, embed_dim)
        self.year_proj = nn.Linear(1, embed_dim)
        self.rating_proj = nn.Linear(1, embed_dim)
        self.duration_proj = nn.Linear(1, embed_dim)
        self.votes_proj = nn.Linear(1, embed_dim)

        n_modalities = 8 + (1 if self.use_cast else 0)
        if self.use_cast:
            self.cast_proj = nn.Linear(n_cast, embed_dim)

        self.fuse = nn.Sequential(
            nn.Linear(n_modalities * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, text_features, features: dict, drop: tuple = (), modality_dropout: float = 0.0) -> torch.Tensor:
        """drop: metadata modalities to zero out (e.g. META for text-only queries).
        modality_dropout: during training, randomly zero each metadata modality per sample."""
        named = [
            ("text", self.text_proj(text_features)),
            ("genres", self.genre_proj(features["genres"])),
            ("director", self.director_proj(self.director_emb(features["director"]))),
            ("writer", self.writer_proj(self.writer_emb(features["writer"]))),
            ("year", self.year_proj(features["year"])),
            ("rating", self.rating_proj(features["rating"])),
            ("duration", self.duration_proj(features["duration"])),
            ("votes", self.votes_proj(features["votes"])),
        ]
        if self.use_cast:
            named.append(("cast", self.cast_proj(features["cast"])))

        parts = []
        for name, p in named:
            if name in drop:
                p = torch.zeros_like(p)
            elif name != "text" and self.training and modality_dropout > 0:
                keep = (torch.rand(p.size(0), 1, device=p.device) > modality_dropout).to(p.dtype)
                p = p * keep
            parts.append(p)
        z = self.fuse(torch.cat(parts, dim=-1))
        return F.normalize(z, p=2, dim=-1)


META = MovieRecommender.META
