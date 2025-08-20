#!/usr/bin/env python3
# soft_labels_vmf_scaling_expressive.py
#
# Goal:
#   Assign probabilistic (soft) cluster labels to ICON features using
#   OBS-derived centroids on the unit sphere, with per-cluster “sharpness”
#   (vMF-like concentration) estimated from OBS compactness.
#
# Pipeline overview:
#   1) Load centroids C (K,D), ICON features X (N,D), and optionally OBS features Y (M,D).
#   2) L2-normalize rows so all vectors lie on the unit sphere.
#   3) From OBS, estimate each cluster's compactness -> sharpness s_k.
#      Intuition: tighter clusters (higher mean cosine to centroid) get larger s_k.
#   4) For ICON features, compute cosine similarities S = X @ C^T.
#   5) Form logits = S * s (and optionally add log priors from OBS class frequencies).
#   6) Softmax(logits) -> probabilities; take argmax for hard labels and record confidence.
#   7) Optionally reject low-confidence assignments with label -1.
#
# Notes:
#   - vMF-like: for unit vectors, cosine similarity is proportional to the log-density of a
#     von Mises–Fisher distribution with concentration κ. We approximate κ per cluster by s_k.
#
# Author: DC + helper

# python soft_labels_vmf_scaling_expressive.py --centroids /p/scratch/exaww/chatterjee1/nn_obs/all_nc/features_obs/obs_final_10_centroids_multiyear.pth --icon_features /p/scratch/exaww/chatterjee1/nn_obs/all_nc/features_icon/trainfeat_new_multiyear.pth --out_dir /p/scratch/exaww/chatterjee1/nn_obs/all_nc/features_icon/ --obs_features /p/scratch/exaww/chatterjee1/nn_obs/all_nc/features_obs/trainfeat_multiyear.pth --reject_threshold 0.0

from __future__ import annotations
import os
import argparse
import numpy as np
import torch
from typing import Tuple

# ----------------------------- Utilities ------------------------------------ #

def as_numpy_f32(x) -> np.ndarray:
    """
    Convert various tensor/array types returned by torch.load(...) to a NumPy float32 array.
    - Accepts: torch.Tensor, np.ndarray, list-like
    - Always returns: np.ndarray(dtype=float32)
    """
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


def l2norm_rows(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Row-wise L2 normalization: each row scaled to unit length.
    Handles zero rows by dividing by max(norm, eps).
    """
    a = np.asarray(a, dtype=np.float32)
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.maximum(norms, eps)


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax along the last dimension (K classes).
    logits: (N, K) real-valued scores
    returns: (N, K) probabilities summing to 1 for each row.
    """
    max_per_row = logits.max(axis=1, keepdims=True)
    z = np.exp(logits - max_per_row)
    z_sum = z.sum(axis=1, keepdims=True)
    return z / np.maximum(z_sum, 1e-30)


def estimate_sharpness_from_obs(
    obs_feats_unit: np.ndarray,   # (M, D) unit-norm
    centroids_unit: np.ndarray,   # (K, D) unit-norm
    clip_min: float = 1.0,
    clip_max: float = 50.0,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate a per-cluster sharpness s_k from OBS compactness.
    Steps:
      - Assign each OBS feature to its nearest centroid (cosine NN on unit sphere).
      - For each cluster k, compute mean cosine m_k of its assigned members to centroid_k.
      - Map m_k -> s_k via s_k = 1 / (1 - m_k), clipped to [clip_min, clip_max].
        Intuition: higher m_k (tighter) => larger s_k (sharper distribution).

    Returns:
      s:  (K,) float32 sharpness per cluster
      pi: (K,) float64 priors (fraction of OBS points in each cluster)
    """
    M, D = obs_feats_unit.shape
    K, D2 = centroids_unit.shape
    assert D == D2, "OBS and centroid dimensionality must match."

    # Cosine similarity on unit sphere is just a dot product
    sim_obs_centroids = obs_feats_unit @ centroids_unit.T  # (M, K)
    obs_assign = sim_obs_centroids.argmax(axis=1)          # (M,)

    # Priors: class frequency on OBS (can be used as log priors later)
    counts = np.bincount(obs_assign, minlength=K).astype(np.float64)
    pi = counts / max(counts.sum(), 1e-12)                 # (K,)

    # Sharpness s_k from compactness
    s = np.ones(K, dtype=np.float32)
    for k in range(K):
        idx_k = np.where(obs_assign == k)[0]
        if idx_k.size == 0:
            # No OBS points assigned: keep s_k at 1.0 (neutral)
            continue
        # Mean cosine of members to their centroid (on unit sphere)
        mean_cos = float((obs_feats_unit[idx_k] @ centroids_unit[k]).mean())
        sk = 1.0 / max(1.0 - mean_cos, eps)  # monotone map; avoid div by zero
        s[k] = np.clip(sk, clip_min, clip_max)

    return s, pi


# ----------------------------- Main procedure -------------------------------- #

def run_soft_labeling(
    centroids_path: str,
    icon_features_path: str,
    out_dir: str,
    obs_features_path: str | None = None,
    batch_size: int = 200_000,
    use_priors: bool = True,
    reject_threshold: float = 0.0,
    clip_min: float = 1.0,
    clip_max: float = 50.0,
) -> None:
    """
    Execute the vMF-like soft labeling pipeline end-to-end.

    Inputs:
      - centroids_path: .pth with OBS centroids C of shape (K, D)
      - icon_features_path: .pth with ICON features X of shape (N, D)
      - obs_features_path (optional): .pth with OBS features Y used during clustering (M, D)
        If provided, used to estimate sharpness s_k and priors pi_k. Otherwise, defaults to s_k=1 and uniform priors.
      - batch_size: number of ICON features processed per batch to bound memory
      - use_priors: if True, add log(pi_k) from OBS to logits
      - reject_threshold: if > 0, set label = -1 when max probability < threshold
      - clip_min/clip_max: bounds for sharpness s_k

    Saves:
      - icon_labels_soft_vmf.pth                (N,) int32 hard labels (-1 if rejected)
      - icon_labels_soft_vmf_confidence.pth     (N,) float32 max class probability
      - obs_sharpness_vmf_s.pth                 (K,) float32 per-cluster sharpness
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1) Load arrays ----------------------------------------------------- #
    print("Loading centroids and features...")
    C = as_numpy_f32(torch.load(centroids_path))           # (K, D)
    X = as_numpy_f32(torch.load(icon_features_path))       # (N, D)
    print(f"  centroids C: {C.shape}, dtype={C.dtype}")
    print(f"  ICON feats X: {X.shape}, dtype={X.dtype}")

    # ---- 2) Normalize to the unit sphere ----------------------------------- #
    print("Normalizing rows to unit L2 norm...")
    C = l2norm_rows(C)                                     # (K, D)
    X = l2norm_rows(X)                                     # (N, D)

    # ---- 3) Estimate sharpness s_k (and priors pi_k) from OBS if available -- #
    if obs_features_path and os.path.exists(obs_features_path):
        print("Estimating per-cluster sharpness from OBS features...")
        OBS = as_numpy_f32(torch.load(obs_features_path))  # (M, D)
        OBS = l2norm_rows(OBS)
        s, pi = estimate_sharpness_from_obs(
            obs_feats_unit=OBS,
            centroids_unit=C,
            clip_min=clip_min,
            clip_max=clip_max
        )
        print(f"  sharpness s_k: min={s.min():.3f}, max={s.max():.3f}, mean={s.mean():.3f}")
        print(f"  priors pi_k:   min={pi.min():.4f}, max={pi.max():.4f}")
    else:
        print("OBS features not provided – using s_k = 1 (uniform sharpness) and uniform priors.")
        s = np.ones(C.shape[0], dtype=np.float32)
        pi = np.ones(C.shape[0], dtype=np.float64) / C.shape[0]

    # Precompute log priors if requested
    log_pi = np.log(np.maximum(pi, 1e-12)).astype(np.float32) if use_priors else None

    # ---- 4) Process ICON features in batches -------------------------------- #
    N, K = X.shape[0], C.shape[0]
    labels = np.empty(N, dtype=np.int32)
    max_prob = np.empty(N, dtype=np.float32)

    print(f"Scoring {N} ICON features against {K} centroids (batch_size={batch_size})...")
    bs = max(1, int(batch_size))

    for start in range(0, N, bs):
        stop = min(start + bs, N)
        Xb = X[start:stop]                                 # (B, D)

        # 4a) Cosine similarities to each centroid (since unit-normalized, dot = cos)
        cos_sim = Xb @ C.T                                 # (B, K)

        # 4b) vMF-like scaling: tighten/loosen logits per cluster
        logits = cos_sim * s[None, :]                      # (B, K)

        # 4c) Optional class priors (from OBS frequencies)
        if log_pi is not None:
            logits = logits + log_pi[None, :]

        # 4d) Softmax -> probabilities
        P = stable_softmax(logits)                         # (B, K)

        # 4e) Hard label + confidence
        lb = P.argmax(axis=1).astype(np.int32)            # (B,)
        conf = P.max(axis=1).astype(np.float32)           # (B,)

        # 4f) Optional rejection by confidence threshold
        if reject_threshold > 0.0:
            reject_mask = conf < float(reject_threshold)
            lb[reject_mask] = -1

        labels[start:stop] = lb
        max_prob[start:stop] = conf

        if (start // bs) % 10 == 0:
            print(f"  processed {stop}/{N}...")

    # ---- 5) Save outputs ---------------------------------------------------- #
    out_labels = os.path.join(out_dir, "icon_labels_soft_vmf.pth")
    out_conf   = os.path.join(out_dir, "icon_labels_soft_vmf_confidence.pth")
    out_s      = os.path.join(out_dir, "obs_sharpness_vmf_s.pth")

    torch.save(labels, out_labels)
    torch.save(max_prob, out_conf)
    torch.save(s, out_s)

    print("Done. Wrote:")
    print(f"  labels     -> {out_labels}")
    print(f"  confidence -> {out_conf}")
    print(f"  sharpness  -> {out_s}")


# ----------------------------- CLI wrapper ----------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Probabilistic labeling of ICON data via vMF-like per-cluster scaling"
    )
    p.add_argument("--centroids", default='/p/scratch/exaww/chatterjee1/nn_obs/all_nc/features_obs/obs_final_10_centroids_multiyear.pth', 
        required=True, help="OBS centroids .pth (K,D)")
    p.add_argument("--icon_features", default = '/p/scratch/exaww/chatterjee1/nn_obs/all_nc/features_icon/trainfeat_new_multiyear.pth', 
        required=True, help="ICON features .pth (N,D)")
    p.add_argument("--out_dir", default = '/p/scratch/exaww/chatterjee1/nn_obs/all_nc/features_icon/', 
        required=True, help="Output directory")
    p.add_argument("--obs_features", default='/p/scratch/exaww/chatterjee1/nn_obs/all_nc/features_obs/trainfeat_multiyear.pth', 
        help="(Optional) OBS features .pth (M,D)")
    p.add_argument("--batch_size", type=int, default=100, help="Batch size for ICON processing")
    p.add_argument("--use_priors", action="store_true", help="Add log priors (from OBS) to logits")
    p.add_argument("--reject_threshold", type=float, default=0.5, help="If >0, label=-1 when max prob < threshold")
    p.add_argument("--clip_min", type=float, default=1.0, help="Min sharpness s_k")
    p.add_argument("--clip_max", type=float, default=50.0, help="Max sharpness s_k")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_soft_labeling(
        centroids_path=args.centroids,
        icon_features_path=args.icon_features,
        out_dir=args.out_dir,
        obs_features_path=(args.obs_features or None),
        batch_size=args.batch_size,
        use_priors=args.use_priors,
        reject_threshold=args.reject_threshold,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
    )
