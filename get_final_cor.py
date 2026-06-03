import time
import numpy as np
from reg_with_scale import reg_with_scale


def get_final_cor(X, Y, distances, scale):
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
    """
    
    tic = time.time()
    
    # Extract distances (second column)
    if distances.ndim > 1:
        distances = distances[:, 1]
    
    print('Size of X:', X.shape)
    
    # Call registration with optimized parameters
    bestS, bestR, best_T = reg_with_scale(
        X, Y, distances, scale
    )
    
    buildGraphTime = time.time() - tic
    print('Running graph time %.3f\n' % buildGraphTime)
    
    return buildGraphTime, bestS, bestR, best_T
