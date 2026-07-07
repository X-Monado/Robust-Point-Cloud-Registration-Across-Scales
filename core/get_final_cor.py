import time
import numpy as np
from .reg_with_scale import reg_with_scale


def get_final_cor(X, Y, distances, scale, **kwargs):
    """
    Final correspondence processing and registration

    Parameters:
    -----------
    X : numpy.ndarray
        Source point cloud, shape (3, N)
    Y : numpy.ndarray
        Target point cloud, shape (3, N)
    distances : numpy.ndarray
        Distances for each correspondence, shape (N, 2)
    scale : float
        Scale factor
    **kwargs : dict
        Additional arguments passed to reg_with_scale:
        - use_triangular : bool
        - triangular_percent : float
        - epsilon_2, epsilon_3 : float
        - random_seed : int
        - return_diagnostics : bool

    Returns:
    --------
    buildGraphTime : float
        Time taken for graph construction
    bestS : float
        Best scale
    bestR : numpy.ndarray
        Best rotation matrix, shape (3, 3)
    best_T : numpy.ndarray
        Best translation vector, shape (3,)
    diagnostics : dict (only if return_diagnostics=True)
    """
    tic = time.time()

    # Extract distances (second column)
    if distances.ndim > 1:
        distances = distances[:, 1]

    print('Size of X:', X.shape)

    # Call registration with optimized parameters
    result = reg_with_scale(X, Y, distances, scale, **kwargs)

    buildGraphTime = time.time() - tic
    print('Running graph time %.3f\n' % buildGraphTime)

    # Handle both 3-tuple and 4-tuple returns
    if len(result) == 4:
        bestS, bestR, best_T, diagnostics = result
        diagnostics['total_time'] = buildGraphTime
        return buildGraphTime, bestS, bestR, best_T, diagnostics
    else:
        bestS, bestR, best_T = result
        return buildGraphTime, bestS, bestR, best_T
