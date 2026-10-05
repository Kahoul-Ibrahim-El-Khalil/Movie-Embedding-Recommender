"""Evaluate the trained recommender against the original MiniLM baseline.

Ground truth (proxy): for a query movie, relevant catalog items are movies
sharing at least one genre with it. This is NOT user-preference ground truth;
it is the only signal available in this repository (no ratings-by-user).

The test split is read from the run directory (as saved by train.py), never
re-split, so evaluation always matches the training run.

The fair comparison is baseline (original pretrained MiniLM) vs trained
text-only (metadata dropped): the full-input column still benefits from
genre inputs that also define the ground truth.

Usage:
    python -m recommender.evaluate --out-dir artifacts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import FeatureEncoder, load_split, positive_mask
from .features import compute_text_features
from .model import META, MovieRecommender


def encode_catalog(model, text_feats, enc_inputs, device, drop: tuple = ()) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(
            text_feats.to(device), {k: v.to(device) for k, v in enc_inputs.items()}, drop=drop
        ).cpu()


def rank_catalog(query_embs: torch.Tensor, catalog_embs: torch.Tensor, exclude_self: bool = True) -> np.ndarray:
    """Return indices that rank each query's catalog most similar first."""
    sims = query_embs @ catalog_embs.T
    if exclude_self and query_embs is catalog_embs:
        sims.fill_diagonal_(-1.0)
    return torch.argsort(sims, dim=1, descending=True).numpy()


def metrics_at_k(ranking: np.ndarray, relevant: np.ndarray, ks=(1, 5, 10)) -> dict:
    """ranking: [Q, C] ranked catalog indices per query. relevant: [Q, C] bool."""
    results = {}
    for k in ks:
        topk = ranking[:, :k]
        hit = 0
        recalls = []
        precisions = []
        ndcgs = []
        reciprocal_ranks = []
        for i in range(len(ranking)):
            rel_total = relevant[i].sum()
            if rel_total == 0:
                continue
            hits = relevant[i][topk[i]]
            hit += int(hits.any())
            recalls.append(hits.sum() / rel_total)
            precisions.append(hits.sum() / k)
            dcg = sum((1.0 / np.log2(r + 2)) for r, h in enumerate(hits) if h)
            ideal = sum(1.0 / np.log2(r + 2) for r in range(min(rel_total, k)))
            ndcgs.append(dcg / ideal if ideal > 0 else 0.0)
            first = np.where(hits)[0]
            reciprocal_ranks.append(1.0 / (first[0] + 1) if len(first) else 0.0)
        n = max(len(precisions), 1)
        results[f"HitRate@{k}"] = hit / n if n else 0.0
        results[f"Recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
        results[f"Precision@{k}"] = float(np.mean(precisions)) if precisions else 0.0
        results[f"NDCG@{k}"] = float(np.mean(ndcgs)) if ndcgs else 0.0
        results[f"MRR@{k}"] = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None, help="unused: the test split is read from the run directory")
    p.add_argument("--out-dir", default="artifacts")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "dml"])
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    config = json.loads((out_dir / "config.json").read_text())
    encoder = FeatureEncoder.load(out_dir / "feature_encoder.pkl")

    test_df = load_split(out_dir, "test")

    from .features import get_device
    device = get_device(args.device)

    # test text encoded with the same encoder used at indexing time
    tm = str(out_dir / "text_encoder") if (out_dir / "text_encoder").exists() else config.get("text_model", "all-MiniLM-L6-v2")
    test_text = compute_text_features(
        test_df,
        model_name=tm,
        cache_path=out_dir / "text_feats_test.pt",
    )
    test_enc = encoder.encode(test_df)
    model = MovieRecommender(
        text_dim=config["text_dim"], n_genres=config["n_genres"], n_directors=config["n_directors"],
        n_writers=config.get("n_writers", 100), n_cast=config.get("n_cast", 0),
        use_cast=config.get("use_cast", False),
        embed_dim=config["embed_dim"], hidden_dim=config["hidden_dim"], dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(out_dir / "model.pt", map_location=device, weights_only=True))
    trained_embs = encode_catalog(model, test_text, test_enc, device)
    trained_text_only = encode_catalog(model, test_text, test_enc, device, drop=META)

    # true baseline: ORIGINAL pretrained encoder, never the fine-tuned one
    base_name = config.get("text_model", "all-MiniLM-L6-v2")
    base_text = compute_text_features(test_df, model_name=base_name, cache_path=out_dir / "text_feats_test_base.pt")
    baseline_embs = F.normalize(base_text, p=2, dim=-1)

    relevant = positive_mask(test_df["genres"].tolist())
    relevant = torch.from_numpy(relevant)

    ranking_trained = rank_catalog(trained_embs, trained_embs)
    ranking_text_only = rank_catalog(trained_text_only, trained_text_only)
    ranking_baseline = rank_catalog(baseline_embs, baseline_embs)

    print("\n=== Test-set retrieval metrics (genre-overlap ground truth) ===")
    trained_metrics = metrics_at_k(ranking_trained, relevant.numpy())
    text_only_metrics = metrics_at_k(ranking_text_only, relevant.numpy())
    baseline_metrics = metrics_at_k(ranking_baseline, relevant.numpy())
    print(f"{'metric':<12} {'baseline MiniLM':>16} {'trained text-only':>18} {'trained full':>14}")
    for k in trained_metrics:
        print(f"{k:<12} {baseline_metrics[k]:>16.4f} {text_only_metrics[k]:>18.4f} {trained_metrics[k]:>14.4f}")

    report = {"baseline": baseline_metrics, "trained_text_only": text_only_metrics, "trained": trained_metrics}
    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_dir / 'evaluation.json'}")


if __name__ == "__main__":
    main()
