# -*- coding: utf-8 -*-
"""
GPU-accelerated versions of the O(N²) bottleneck functions.

Bottleneck analysis (KITTI pair, N=5000 matches):
  - line_vectors:       1.88s  (Python for-loop, 12.5M vectors)
  - scale_vector rest:  3.60s  (distance/filter on 12.5M vectors)
  - lv_map_to_pt:       1.11s  (Counter-based point selection)
  - triangular:         0.59s  (O(N²) pairwise distances)
  - IRLS:               0.89s  (small 3×3 SVD — kept on CPU)

This module provides drop-in GPU replacements using PyTorch CUDA tensors.
IRLS is left on CPU because it operates on 3×3 matrices (too small for GPU).
"""
import torch
import numpy as np


def _to_torch(X, device='cuda'):
    """Convert numpy (3, N) array to torch tensor on GPU."""
    if isinstance(X, torch.Tensor):
        return X.to(device)
    return torch.from_numpy(np.ascontiguousarray(X)).to(device)


def _to_numpy(t):
    """Convert torch tensor back to numpy."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t


def line_vectors_gpu(X_t, need_map=True):
    """
    GPU line vector computation — replaces the Python for-loop in line_vectors.py.

    Generates all N*(N-1)/2 line vectors X[:,i] - X[:,j] for i < j using
    triu_indices, which produces the same ordering as the original for-loop.

    Parameters
    ----------
    X_t : torch.Tensor, shape (3, N) on GPU
    need_map : bool — if True, also return index map

    Returns
    -------
    LV : torch.Tensor (3, M)   where M = N*(N-1)/2
    ID : torch.Tensor (2, M)   [optional] row 0 = i, row 1 = j
    """
    N = X_t.shape[1]
    idx_i, idx_j = torch.triu_indices(N, N, offset=1, device=X_t.device)
    LV = X_t[:, idx_i] - X_t[:, idx_j]  # (3, M)
    if need_map:
        ID = torch.stack([idx_i, idx_j])  # (2, M)
        return LV, ID
    return LV


def scale_vector_gpu(X_t, Y_t, bound, length_bound):
    """
    GPU scale vector computation — replaces scale_vector.py.

    Computes scale ratios between all pairs of line vectors with length filtering.

    Parameters
    ----------
    X_t, Y_t : torch.Tensor (3, N) on GPU
    bound : float — noise bound
    length_bound : float — minimum line vector length

    Returns
    -------
    Sxy : torch.Tensor (M,) — scale ratios D_ylv / D_xlv
    Snoise : torch.Tensor (M,) — noise scales bound / D_xlv
    map_arr : torch.Tensor (2, M) — index map (i, j) into original points
    """
    X_lv, map_arr = line_vectors_gpu(X_t, need_map=True)
    Y_lv = line_vectors_gpu(Y_t, need_map=False)

    D_xlv = torch.norm(X_lv, dim=0)  # (M,)
    D_ylv = torch.norm(Y_lv, dim=0)  # (M,)

    # Length filtering: discard short line vectors
    mask = (D_xlv > length_bound) & (D_ylv > length_bound)

    D_xlv = D_xlv[mask]
    D_ylv = D_ylv[mask]
    map_arr = map_arr[:, mask]

    eps = torch.finfo(torch.float32).eps
    Sxy = D_ylv / (D_xlv + eps)
    Snoise = bound / (D_xlv + eps)

    return Sxy, Snoise, map_arr


def lv_map_to_pt_gpu(X_t, Y_t, map_arr, inliers, percent, distances_t):
    """
    GPU point selection from line-vector inliers — replaces lv_map_to_pt.py.

    Uses torch.bincount instead of Python Counter for O(N) point counting.

    Parameters
    ----------
    X_t, Y_t : torch.Tensor (3, N) on GPU
    map_arr : torch.Tensor (2, M) — index map
    inliers : array-like — indices of inlier line vectors
    percent : float — cumulative count percentile
    distances_t : torch.Tensor (N,) or (N, 2) on GPU

    Returns
    -------
    Xin, Yin : numpy (3, K)
    distance : numpy (K,)
    inlierPTS : numpy (K,)
    """
    inliers_t = torch.as_tensor(inliers, device=map_arr.device).long()
    map_in = map_arr[:, inliers_t]  # (2, K)

    # Count occurrences of each point index using bincount (fast on GPU)
    all_indices = map_in.flatten()
    N = X_t.shape[1]
    counts = torch.bincount(all_indices, minlength=N)
    counts[counts == 0] = -1  # mark zeros so they sort to the bottom
    counts[counts == -1] = 0  # set back to 0 (won't be selected)

    # Actually, we need: sort by count descending, select top by cumulative percentile
    # Use a simpler approach: argsort on negative counts
    counts_clean = torch.bincount(all_indices, minlength=N)
    # Sort indices by count descending
    sorted_indices = torch.argsort(counts_clean, descending=True)
    sorted_counts = counts_clean[sorted_indices]

    # Remove zero-count entries
    nonzero = sorted_counts > 0
    sorted_indices = sorted_indices[nonzero]
    sorted_counts = sorted_counts[nonzero]

    if len(sorted_indices) == 0:
        return np.array([]).reshape(3, 0), np.array([]).reshape(3, 0), np.array([]), np.array([])

    cumsum = torch.cumsum(sorted_counts, dim=0)
    total = cumsum[-1].item()
    threshold = (percent / 100.0) * total

    p_idx = cumsum < threshold
    # Ensure at least 1 point
    if not p_idx.any():
        p_idx[0] = True

    inlierPTS_t = sorted_indices[p_idx]

    # Sort by distance (descending — matches original: smaller feature distance = keep)
    if distances_t.dim() == 2 and distances_t.shape[1] == 2:
        dist_1d = distances_t[:, 1]
    else:
        dist_1d = distances_t

    distance_pre = dist_1d[inlierPTS_t]
    sorted_idx = torch.argsort(distance_pre, descending=True)
    distance = distance_pre[sorted_idx]
    inlierPTS_t = inlierPTS_t[sorted_idx]

    Xin = _to_numpy(X_t[:, inlierPTS_t])
    Yin = _to_numpy(Y_t[:, inlierPTS_t])
    distance = _to_numpy(distance)
    inlierPTS = _to_numpy(inlierPTS_t)

    return Xin, Yin, distance, inlierPTS


def compute_vertex_degrees_gpu(X_t, Y_t, s, epsilon_2, epsilon_3, anchor_idx):
    """
    GPU triangular consistency vertex degree computation.

    Uses torch.cdist for pairwise distances instead of numpy broadcasting.

    Parameters
    ----------
    X_t, Y_t : torch.Tensor (3, N) on GPU
    s : float — scale
    epsilon_2, epsilon_3 : float — compatibility thresholds
    anchor_idx : int — anchor point index

    Returns
    -------
    degrees : numpy (N,)
    edge_mask_upper : numpy (N, N) bool
    """
    N = X_t.shape[1]

    # Anchor-to-point distances
    x_a = X_t[:, anchor_idx:anchor_idx + 1]  # (3, 1)
    y_a = Y_t[:, anchor_idx:anchor_idx + 1]
    D_xa = torch.norm(X_t - x_a, dim=0)  # (N,)
    D_ya = torch.norm(Y_t - y_a, dim=0)

    c12_mask = torch.abs(D_ya - s * D_xa) < epsilon_2

    # Pairwise distances via cdist (N×N on GPU)
    D_x = torch.cdist(X_t.T, X_t.T)  # (N, N)
    D_y = torch.cdist(Y_t.T, Y_t.T)

    c3_mask = torch.abs(D_y - s * D_x) < epsilon_3

    c12_outer = torch.outer(c12_mask, c12_mask)
    c12_outer.fill_diagonal_(False)
    c12_outer[anchor_idx, :] = False
    c12_outer[:, anchor_idx] = False

    edge_mask = c12_outer & c3_mask
    edge_mask_upper = torch.triu(edge_mask, diagonal=1)

    degrees = edge_mask_upper.sum(dim=0) + edge_mask_upper.sum(dim=1)

    return _to_numpy(degrees), _to_numpy(edge_mask_upper)


def filter_by_triangular_consistency_gpu(X_t, Y_t, s, epsilon_2, epsilon_3, percent=50):
    """
    GPU triangular consistency filtering — replaces filter_by_triangular_consistency.

    Parameters
    ----------
    X_t, Y_t : torch.Tensor (3, N) on GPU
    s : float, epsilon_2/epsilon_3 : float, percent : float

    Returns
    -------
    inlier_indices : numpy (K,)
    degrees : numpy (N,)
    """
    N = X_t.shape[1]
    num_anchors = min(10, N)

    # Use point 0 as initial anchor to select top-K anchors
    degrees_np, _ = compute_vertex_degrees_gpu(X_t, Y_t, s, epsilon_2, epsilon_3, 0)
    sorted_indices = np.argsort(-degrees_np)
    anchor_indices = sorted_indices[:num_anchors]

    adjacency = np.zeros((N, N), dtype=bool)
    degrees = np.zeros(N, dtype=int)

    for a in anchor_indices:
        _, edge_mask = compute_vertex_degrees_gpu(X_t, Y_t, s, epsilon_2, epsilon_3, int(a))
        new_edges = edge_mask & ~adjacency
        adjacency |= new_edges
        degrees += np.sum(new_edges, axis=0) + np.sum(new_edges, axis=1)

    # Sort by degree (descending)
    sorted_indices = np.argsort(-degrees)
    sorted_degrees = degrees[sorted_indices]

    cumsum = np.cumsum(sorted_degrees)
    total = cumsum[-1] if len(cumsum) > 0 and cumsum[-1] > 0 else 0

    if total == 0:
        return np.arange(N), degrees

    threshold = (percent / 100.0) * total
    p_idx = cumsum < threshold

    if np.sum(p_idx) < 3:
        p_idx[:min(5, N)] = True

    inlier_indices = sorted_indices[p_idx]
    return inlier_indices, degrees
