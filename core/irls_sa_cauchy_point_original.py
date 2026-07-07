"""
Original IRLS with Simulated Annealing and Cauchy distribution.

Recovered via decompilation from the March 26, 2026 pyc file
(__pycache__/irls_sa_cauchy_point.cpython-36.pyc) that produced the
original 73.2% success rate on KITTI-FPFH (555 pairs).

Key differences from the simplified version (irls_sa_cauchy_point_simple.py):
  1. div = 1.2 (faster temperature cooling; simplified uses 1.12)
  2. u1/u2 computed WITHOUT np.abs (simplified uses np.abs)
  3. Convergence check: early break when weights stabilize (< 1e-06)
  4. Post-processing: weight thresholding at 0.7 + final re-estimation
     if more than 10 high-confidence points remain
  5. No best_R/best_t tracking (dead code in original; simplified tracks best)
"""

import numpy as np
from sa_cauchy_point import sa_cauchy_point


def irls_sa_cauchy_point_original(x, y, iter_num, sigma):
    """
    Iteratively reweighted least squares with simulated annealing and Cauchy distribution

    Parameters:
    -----------
    x : numpy.ndarray
        Source point cloud, shape (3, N)
    y : numpy.ndarray
        Target point cloud, shape (3, N)
    iter_num : int
        Number of iterations
    sigma : float
        Noise bound

    Returns:
    --------
    R : numpy.ndarray
        Estimated rotation matrix, shape (3, 3)
    t : numpy.ndarray
        Estimated translation vector, shape (3,)
    W : numpy.ndarray
        Final weights, shape (3, N)
    """
    W = np.ones((3, x.shape[1]))
    w = W[0, :]
    u1 = np.max(x[:])
    u2 = np.min(x[:])
    u = (u1 - u2) ** 2
    div = 1.2
    max_cout = 0
    best_R = np.eye(3)
    best_t = np.zeros(3)

    for i in range(iter_num):
        R, t, W = sa_cauchy_point(x, y, u, W[0, :])
        if np.max(np.max(np.abs(w - W[0, :]))) < 1e-06:
            break
        w = W[0, :]
        if u > (3 * sigma) ** 2:
            u = u / div

    # Post-processing: threshold weights and re-estimate with high-confidence points
    W[(0, W[0, :] < 0.7)] = 0
    w = W[0, :]
    if np.sum(w) > 10:
        R, t, W = sa_cauchy_point(x, y, u, w)

    return (R, t, W)
