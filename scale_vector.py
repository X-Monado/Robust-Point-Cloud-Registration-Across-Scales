import numpy as np
from line_vectors import line_vectors


def scale_vector(X, Y, bound, length_bound):
    """
    Compute scale vectors from point correspondences
    
    Parameters:
    -----------
    X : numpy.ndarray
        Source point cloud, shape (3, N)
    Y : numpy.ndarray
        Target point cloud, shape (3, N)
    bound : float
        Noise bound
    length_bound : float
        Length bound for filtering
    
    Returns:
    --------
    Sxy : numpy.ndarray
        Scale ratios between line vectors, shape (M,)
    Snoise : numpy.ndarray
        Noise scales for each ratio, shape (M,)
    X_lv : numpy.ndarray
        Line vectors from source, shape (3, M)
    Y_lv : numpy.ndarray
        Line vectors from target, shape (3, M)
    map : numpy.ndarray
        Index map, shape (2, M)
    """
    length_bound = 0
    X_lv, map = line_vectors(X, 1)
    Y_lv = line_vectors(Y, 0)
    
    D_xlv = np.sqrt(np.sum(X_lv**2, axis=0))
    D_ylv = np.sqrt(np.sum(Y_lv**2, axis=0))
    
    idx1 = (D_xlv <= length_bound)
    idx2 = (D_ylv <= length_bound)
    idx = idx1 | idx2
    
    D_xlv = D_xlv[~idx]
    D_ylv = D_ylv[~idx]
    map = map[:, ~idx]
    X_lv = X_lv[:, ~idx]
    Y_lv = Y_lv[:, ~idx]
    
    eps = np.finfo(float).eps
    Sxy = D_ylv / (D_xlv + eps)
    Snoise = bound / (D_xlv + eps)
    
    return Sxy, Snoise, X_lv, Y_lv, map
