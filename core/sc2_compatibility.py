# -*- coding: utf-8 -*-
"""
Scale-aware Second-order Spatial Compatibility (SC2) graph filtering.

Implements the SC2 compatibility matrix used in MAC (Zhang et al.) and SC2-PCR
(Chen et al.):
    SCG = (FCG @ FCG) * FCG   (element-wise product)

Key adaptation: scale-aware cross-distance  cross_dist = |D_y - s * D_x|
(MAC/SC2-PCR assume s=1 same-scale; our cross-scale pipeline estimates s first).

The module mirrors the interface of triangular_consistency.py so that SC2 can
be used as a drop-in alternative graph-filtering stage in reg_with_scale.py.
"""
import numpy as np


def _pairwise_distance(P):
    """Pairwise Euclidean distance matrix of columns of P (3, N) -> (N, N)."""
    # ‖p_i - p_j‖ via (p_i^2 - 2 p_i·p_j + p_j^2)
    sq = np.sum(P * P, axis=0)          # (N,)
    d2 = sq[:, None] - 2.0 * (P.T @ P) + sq[None, :]
    np.maximum(d2, 0, out=d2)
    return np.sqrt(d2)


def build_sc2_matrix(X, Y, s, d_thre, soft=True, hard_thresh=0.99, max_n=800):
    """
    Build the second-order spatial compatibility matrix SCG.

    Parameters
    ----------
    X : (3, N) source correspondence points
    Y : (3, N) target correspondence points
    s : float, estimated scale (cross_dist = |D_y - s*D_x|)
    d_thre : float, first-order compatibility threshold (epsilon_3)
    soft : bool, if True use MAC-style soft FCG = clamp(1 - cd^2/d_thre^2, 0)
           followed by hard binarisation at `hard_thresh`; if False use a plain
           hard binary FCG = (cross_dist < d_thre).
    hard_thresh : float, binarisation threshold for the soft FCG (MAC uses 0.99)
    max_n : int, subsample to at most this many points to bound O(N^2) memory.

    Returns
    -------
    SCG : (M, M) second-order compatibility matrix (M <= N)
    idx_keep : (M,) indices of the retained (subsampled) points
    degrees : (M,) row-sum degree in the SCG graph
    """
    n = X.shape[1]
    # Subsample if too large (same protection as triangular max_pts_for_tri)
    if n > max_n:
        rng = np.random.default_rng(42)
        idx_keep = rng.choice(n, max_n, replace=False)
        idx_keep.sort()
        Xs = X[:, idx_keep]
        Ys = Y[:, idx_keep]
    else:
        idx_keep = np.arange(n)
        Xs = X
        Ys = Y

    # Pairwise distances (computed once — unlike triangular which recomputes per anchor)
    D_x = _pairwise_distance(Xs)
    D_y = _pairwise_distance(Ys)

    # Scale-aware cross distance
    cross_dist = np.abs(D_y - s * D_x)

    # First-order compatibility graph FCG
    if soft:
        FCG = np.clip(1.0 - cross_dist ** 2 / (d_thre ** 2), 0.0, None)
        np.fill_diagonal(FCG, 0.0)
        FCG[FCG < hard_thresh] = 0.0      # MAC-style hard binarisation
        FCG[FCG > 0] = 1.0
    else:
        FCG = (cross_dist < d_thre).astype(np.float64)
        np.fill_diagonal(FCG, 0.0)

    # Second-order compatibility: SCG = (FCG @ FCG) * FCG
    SCG = (FCG @ FCG) * FCG

    # Degree = row sum of SCG (number/weight of second-order compatible neighbours)
    degrees = SCG.sum(axis=1)

    return SCG, idx_keep, degrees


def filter_by_sc2(X, Y, s, d_thre, percent=50, soft=True, hard_thresh=0.99,
                  max_n=800):
    """
    Filter correspondences by SC2 vertex degree (cumulative-percentile retention).

    Mirrors filter_by_triangular_consistency / lv_map_to_pt: sort by degree
    descending, retain the top `percent`% by cumulative degree.

    Parameters
    ----------
    X, Y : (3, N) correspondence points
    s : float, estimated scale
    d_thre : float, first-order compatibility threshold
    percent : float, cumulative-degree percentile to retain (0-100)
    soft, hard_thresh, max_n : see build_sc2_matrix

    Returns
    -------
    inlier_indices : (K,) indices (into the original N points) retained
    degrees : (M,) degree of each retained subsampled point
    """
    n = X.shape[1]
    if n < 6:
        return np.arange(n), np.ones(n)

    SCG, idx_keep, degrees = build_sc2_matrix(
        X, Y, s, d_thre, soft=soft, hard_thresh=hard_thresh, max_n=max_n)

    M = len(idx_keep)
    # Sort by degree descending
    order = np.argsort(-degrees)
    deg_sorted = degrees[order]
    total_deg = deg_sorted.sum()
    if total_deg <= 0:
        # Degenerate: no compatible edges, fall back to all points
        return idx_keep, degrees

    # Cumulative-degree percentile retention (same scheme as lv_map_to_pt / triangular)
    cum = np.cumsum(deg_sorted) / total_deg
    cutoff = np.searchsorted(cum, percent / 100.0) + 1
    cutoff = max(min(cutoff, M), min(5, M))   # keep at least min(5,M)

    keep_local = order[:cutoff]
    inlier_indices = idx_keep[keep_local]
    return inlier_indices, degrees[keep_local]
