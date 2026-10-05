"""3D PCA visualization of the learned movie embedding space.

Usage:
    python -m recommender.visualize --out-dir runs/minilm [--n 300] [--html viz.html]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA


def visualize(out_dir: str | Path = "artifacts", n: int = 300, html: str | None = None,
              title: str | None = None, seed: int = 0):
    out_dir = Path(out_dir)
    embs = torch.load(out_dir / "catalog_embeddings.pt", weights_only=True).numpy()
    catalog = pd.read_csv(out_dir / "catalog.csv")

    if len(embs) != len(catalog):
        raise RuntimeError(
            f"catalog_embeddings.pt has {len(embs)} rows but catalog.csv has {len(catalog)}. "
            f"The index is stale; rebuild it with the 'index' command."
        )
    if len(embs) < 3:
        raise RuntimeError("Need at least 3 movies for a 3D PCA plot.")

    if len(embs) > n:
        idx = np.random.default_rng(seed).choice(len(embs), n, replace=False)
        embs, catalog = embs[idx], catalog.iloc[idx]
    catalog = catalog.reset_index(drop=True)

    pca = PCA(n_components=3)
    reduced = pca.fit_transform(embs)
    explained = float(pca.explained_variance_ratio_.sum())

    primary = (
        catalog["genres"].fillna("").astype(str).str.split(",").str[0].str.strip().replace("", "Unknown")
    )

    import plotly.graph_objects as go

    fig = go.Figure()
    for genre in sorted(primary.unique()):
        m = (primary == genre).to_numpy()
        sub = catalog[m]
        fig.add_trace(
            go.Scatter3d(
                x=reduced[m, 0], y=reduced[m, 1], z=reduced[m, 2],
                mode="markers",
                name=genre,
                marker=dict(size=4, opacity=0.8),
                hovertext=[
                    f"{t} ({int(y)})" if pd.notna(y) else str(t)
                    for t, y in zip(sub["title"], sub["year"])
                ],
                hoverinfo="text",
            )
        )
    fig.update_layout(
        title=title or f"Learned Movie Embedding Space (PCA → 3D, {explained:.0%} variance, colour = first genre)",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        width=900, height=750,
    )
    if html:
        fig.write_html(html)
        print(f"Wrote {html}")
    fig.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="artifacts")
    p.add_argument("--n", type=int, default=300, help="Max movies to plot")
    p.add_argument("--html", default=None, help="Also save to this HTML file")
    args = p.parse_args()
    visualize(out_dir=args.out_dir, n=args.n, html=args.html)


if __name__ == "__main__":
    main()
