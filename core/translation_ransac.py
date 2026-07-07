# -*- coding: utf-8 -*-
"""
1-point translation RANSAC given a known rotation.

When the rotation R is reliable but the translation t is biased (e.g. due to
contaminated IRLS weights in low-inlier-ratio outdoor scenes), this module
re-estimates t by finding the densest cluster among per-match translation
candidates  t_i = Y_i - R @ X_i.

This is far more robust than the weighted-centroid estimate used in
sa_cauchy_point when the inlier ratio is low, because it explicitly searches
for the translation mode instead of averaging over potentially-biased weights.
"""
import numpy as np


def translation_ransac_1pt(Xs, Y, R, threshold, max_n=1000, rng_seed=42):
    """
    1-point RANSAC for translation given a known rotation.

    Parameters
    ----------
    Xs : (3, N) source points AFTER scale application (i.e. bestS * X)
    Y  : (3, N) target points
    R  : (3, 3) known rotation
    threshold : float — inlier distance threshold for translation candidates
    max_n : int — subsample cap for speed (O(max_n^2))
    rng_seed : int

    Returns
    -------
    t : (3,) estimated translation
    inlier_count : int — number of inliers at the best translation
    inlier_ratio : float — inlier_count / n_used
    """
    n = Xs.shape[1]
    if n < 4:
        return np.zeros(3), 0, 0.0

    if n > max_n:
        idx = np.random.RandomState(rng_seed).choice(n, max_n, replace=False)
        Xs = Xs[:, idx]
        Y = Y[:, idx]
        n = max_n

    # Per-match translation candidates: t_i = Y_i - R @ Xs_i
    t_cands = Y - R @ Xs  # (3, n)
    tT = t_cands.T  # (n, 3)

    # Vectorized pairwise distances in chunks: dists[i, j] = |t_i - t_j|
    best_count = 0
    best_inliers = None
    best_t = np.zeros(3)
    chunk = 256
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        diff = tT[start:end, None, :] - tT[None, :, :]  # (chunk, n, 3)
        dists = np.linalg.norm(diff, axis=2)  # (chunk, n)
        counts = np.sum(dists < threshold, axis=1)  # (chunk,)
        local_best = int(np.argmax(counts))
        if counts[local_best] > best_count:
            best_count = int(counts[local_best])
            best_inliers = dists[local_best] < threshold
            best_t = t_cands[:, start + local_best]

    # Refine: mean of inliers (robust central estimate of the cluster)
    if best_inliers is not None and best_count >= 3:
        best_t = np.mean(t_cands[:, best_inliers], axis=1)

    return best_t, best_count, best_count / n
