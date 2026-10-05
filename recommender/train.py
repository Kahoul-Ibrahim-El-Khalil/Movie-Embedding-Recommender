"""Training entry point.

Usage:
    python -m recommender.train --data Raw/16k_Movies.csv --epochs 30

Objective: multi-positive InfoNCE over in-batch negatives.

    positives(i) = movies in the same batch sharing >= 1 genre with i
                   (or Jaccard-weighted with --jaccard)
    negatives    = every other movie in the batch

Movies with no positive in the batch contribute no term. This directly
optimizes the embedding space for recommendation-style retrieval: similar-genre
movies are pulled together, unrelated movies are pushed apart.

Genre inputs get modality dropout during training so the model cannot just
memorize the genre label that also defines the positives; text-only queries
(see recommend --description) then work at inference time.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .dataset import FeatureEncoder, load_movies, train_val_test_split
from .features import compute_text_features, encode_with_grad, get_device
from .model import MovieRecommender


def batch_positives(genres: torch.Tensor, jaccard: bool = False) -> torch.Tensor:
    """genres: [B, G] multi-hot float. Returns [B, B] float weights with a zero diagonal.
    Binary: 1 when the pair shares a genre. Jaccard: |A∩B| / |A∪B|."""
    inter = genres @ genres.T
    if jaccard:
        sizes = genres.sum(dim=1)
        union = sizes[:, None] + sizes[None, :] - inter
        w = inter / union.clamp(min=1.0)
    else:
        w = (inter > 0).float()
    eye = torch.eye(genres.size(0), dtype=torch.bool).to(w.device)  # built on CPU for DirectML
    return w.masked_fill(eye, 0.0)


def info_nce_multi_positive(embeddings: torch.Tensor, weights: torch.Tensor, temperature: float) -> torch.Tensor:
    """embeddings: [B, D] L2-normalized. weights: [B, B] float >= 0, zero diagonal."""
    sim = embeddings @ embeddings.T / temperature
    B = embeddings.size(0)
    eye = torch.eye(B, dtype=torch.bool).to(sim.device)
    sim = sim.masked_fill(eye, -1e9)                      # exclude self from the denominator
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    w_sum = weights.sum(dim=1)
    valid = w_sum > 0                                     # anchors with at least one positive
    if not valid.any():
        return embeddings.sum() * 0.0
    per_anchor = -(weights * log_prob).sum(dim=1) / w_sum.clamp(min=1e-12)
    return per_anchor[valid].mean()


def run_epoch(model, loader, encoder_inputs, optimizer, temperature, device, train: bool,
              text_feats=None, st_model=None, st_texts=None,
              modality_dropout: float = 0.0, jaccard: bool = False):
    model.train(train)
    if st_model is not None:
        st_model.train(train)
    total, count = 0.0, 0
    with torch.set_grad_enabled(train):
        for (idx,) in loader:
            if st_model is not None:
                feats = encode_with_grad(st_model, [st_texts[int(i)] for i in idx], device)
            else:
                feats = text_feats[idx].to(device)
            inputs = {k: v[idx].to(device) for k, v in encoder_inputs.items()}
            z = model(feats, inputs, modality_dropout=modality_dropout if train else 0.0)
            pos = batch_positives(encoder_inputs["genres"][idx], jaccard).to(device)
            loss = info_nce_multi_positive(z, pos, temperature)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(idx)
            count += len(idx)
    return total / max(count, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to movie CSV (data.csv or 16k format)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--text-lr", type=float, default=2e-5, help="LR for the sentence encoder when fine-tuning")
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--modality-dropout", type=float, default=0.3, help="per-sample prob. of zeroing each metadata input")
    p.add_argument("--jaccard", action="store_true", help="weight positives by genre Jaccard instead of any-overlap")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--text-model", default="all-MiniLM-L6-v2")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "dml"])
    p.add_argument("--finetune-text", action="store_true", help="Fine-tune MiniLM jointly (slower, better)")
    p.add_argument("--out-dir", default="artifacts")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_movies(args.data)
    print(f"Loaded {len(df)} movies from {args.data}")
    train_df, val_df, test_df = train_val_test_split(df, seed=args.seed)
    print(f"Split: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")

    encoder = FeatureEncoder().fit(train_df)
    enc_inputs = encoder.encode(train_df)
    val_enc_inputs = encoder.encode(val_df)

    device = get_device(args.device)

    if args.finetune_text:
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer(args.text_model).to(device)
        st_train_texts = train_df["description"].tolist()
        st_val_texts = val_df["description"].tolist()
        text_train = text_val = None
        train_text_dim = st_model.get_sentence_embedding_dimension()
    else:
        st_model = st_train_texts = st_val_texts = None
        slug = args.text_model.replace("/", "_")
        text_train = compute_text_features(train_df, model_name=args.text_model, cache_path=out_dir / f"text_feats_train_{slug}.pt")
        text_val = compute_text_features(val_df, model_name=args.text_model, cache_path=out_dir / f"text_feats_val_{slug}.pt")
        compute_text_features(test_df, model_name=args.text_model, cache_path=out_dir / f"text_feats_test_{slug}.pt")
        train_text_dim = text_train.shape[1]

    train_loader = DataLoader(TensorDataset(torch.arange(len(train_df))), batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(torch.arange(len(val_df))), batch_size=args.batch_size)

    model = MovieRecommender(
        text_dim=train_text_dim,
        n_genres=encoder.n_genres,
        n_directors=encoder.n_directors,
        n_writers=encoder.n_writers,
        n_cast=encoder.n_cast,
        use_cast=encoder.n_cast > 0,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    groups = [{"params": list(model.parameters()), "lr": args.lr}]
    if st_model is not None:
        groups.append({"params": list(st_model.parameters()), "lr": args.text_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)

    best_val = float("inf")
    patience, since_best = 5, 0
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, enc_inputs, optimizer, args.temperature, device, train=True,
                               text_feats=text_train, st_model=st_model, st_texts=st_train_texts,
                               modality_dropout=args.modality_dropout, jaccard=args.jaccard)
        val_loss = run_epoch(model, val_loader, val_enc_inputs, None, args.temperature, device, train=False,
                             text_feats=text_val, st_model=st_model, st_texts=st_val_texts,
                             jaccard=args.jaccard)
        print(f"epoch {epoch:3d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}")
        if val_loss < best_val:
            best_val, since_best = val_loss, 0
            model.to("cpu")
            torch.save(model.state_dict(), out_dir / "model.pt")
            model.to(device)
            if st_model is not None:
                st_model.to("cpu")
                st_model.save(str(out_dir / "text_encoder"))
                st_model.to(device)
        else:
            since_best += 1
            if since_best >= patience:
                print("early stopping")
                break

    encoder.save(out_dir / "feature_encoder.pkl")
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        part.assign(
            genres=part["genres"].apply(",".join),
            cast=part["cast"].apply(",".join),
        ).to_csv(out_dir / f"{name}_movies.csv", index=False)

    config = vars(args)
    config["n_genres"] = encoder.n_genres
    config["n_directors"] = encoder.n_directors
    config["n_writers"] = encoder.n_writers
    config["n_cast"] = encoder.n_cast
    config["use_cast"] = encoder.n_cast > 0
    config["text_dim"] = int(train_text_dim)
    config["finetune_text"] = bool(args.finetune_text)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Saved model, encoder, splits and config to {out_dir}/")


if __name__ == "__main__":
    main()
