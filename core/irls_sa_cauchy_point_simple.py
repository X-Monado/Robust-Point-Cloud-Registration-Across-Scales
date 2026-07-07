# -*- coding: utf-8 -*-
"""
Simplified IRLS with Simulated Annealing and Cauchy distribution.

This is the legacy version (pre-Jul-2025 rewrite) that uses direct weight
updates without blending or adaptive masking. It is critical for low-inlier-
ratio scenarios (e.g., KITTI-FPFH ~0.3% inlier ratio) where the v3 IRLS's
weight blending (alpha=0.6) prevents proper convergence.

Verified: reproduces the original 73% registration rate on KITTI-FPFH (555 pairs)
that the v3 IRLS (31%) could not achieve.
"""
import numpy as np
from .sa_cauchy_point import sa_cauchy_point


def irls_sa_cauchy_point_simple(x, y, iter_num, sigma):
    """
    Simplified IRLS — direct Cauchy weight update, no blending, no refinement.

    Parameters:
    -----------
    x : numpy.ndarray (3, N) - Source point cloud
    y : numpy.ndarray (3, N) - Target point cloud
    iter_num : int - Number of iterations
    sigma : float - Noise bound (2*sqrt(3)*noise)

    Returns:
    --------
    R : numpy.ndarray (3, 3) - Estimated rotation matrix
    t : numpy.ndarray (3,) - Estimated translation vector
    W : numpy.ndarray (3, N) - Final weights
    """
    n_points = x.shape[1]
    w = np.ones(n_points)

    u1 = np.max(np.abs(x[:]))
    u2 = np.min(np.abs(x[:]))
    u = (u1 - u2) ** 2

    div = 1.12
    best_R = np.eye(3)
    best_t = np.zeros(3)
    best_error = float('inf')

    for i in range(iter_num):
        R, t, W_new = sa_cauchy_point(x, y, u, w)

        Bfit = np.dot(R, x) + t.reshape(-1, 1)
        errors = np.sqrt(np.sum((Bfit - y) ** 2, axis=0))

        # Direct weight update (NO blending, NO adaptive mask)
        w = W_new[0, :]

        # Annealing
        min_u = (3 * sigma) ** 2
        if u > min_u:
            u = u / div

        # Track best by weighted error
        weighted_error = np.mean(w * errors)
        if weighted_error < best_error:
            best_error = weighted_error
            best_R = R.copy()
            best_t = t.copy()

    # Return weights in (3, N) format for compatibility
    W_out = np.vstack([w, w, w])
    return best_R, best_t, W_out
