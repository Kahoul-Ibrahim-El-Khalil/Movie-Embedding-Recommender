"""Compute and cache frozen MiniLM text features for movie descriptions.

The pretrained language model is used only as a *fixed* text feature extractor.
All recommendation-specific adaptation happens in `MovieRecommender`, which is
trained with a contrastive objective.
"""

from __future__ import annotations

import hashlib

from pathlib import Path

import numpy as np
import pandas as pd
import torch

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def _cache_file(cache_path, model_name: str, texts: list[str]) -> Path:
    key = model_name + "\x00" + "\x00".join(texts)
    mp = Path(model_name)
    if mp.exists():  # fine-tuned encoder dir: invalidate when it is retrained
        key += f"\x00{mp.stat().st_mtime}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    p = Path(cache_path)
    return p.with_name(f"{p.stem}-{h}{p.suffix}")


def compute_text_features(
    df: pd.DataFrame,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 64,
    cache_path: str | Path | None = None,
) -> torch.Tensor:
    texts = df["description"].fillna("").tolist()
    cache_file = _cache_file(cache_path, model_name, texts) if cache_path is not None else None
    if cache_file is not None and cache_file.exists():
        return torch.load(cache_file, weights_only=True)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    feats = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    tensor = torch.from_numpy(np.asarray(feats, dtype=np.float32))

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, cache_file)
    return tensor


def encode_with_grad(st, texts, device, batch_size: int = 32) -> torch.Tensor:
    """Encode texts with a SentenceTransformer, keeping gradients (fine-tuning)."""
    outs = []
    for i in range(0, len(texts), batch_size):
        features = st.tokenize(texts[i : i + batch_size])
        features = {k: v.to(device) for k, v in features.items() if hasattr(v, "to")}
        outs.append(st(features)["sentence_embedding"])
    return torch.cat(outs, dim=0)


def get_device(name: str):
    """'cpu' | 'cuda' | 'dml' (AMD DirectML)."""
    if name == "dml":
        try:
            import torch_directml
        except ImportError as e:
            raise RuntimeError("Install DirectML support first: pip install torch-directml") from e
        return torch_directml.device()
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available in this torch build (CPU-only or ROCm needed).")
    return torch.device(name)
