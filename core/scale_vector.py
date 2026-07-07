import numpy as np
from .line_vectors import line_vectors


def scale_vector(X, Y, bound, length_bound):
    """
    Compute scale vectors from point correspondences with length filtering
    
    Parameters:
    -----------
    X : numpy.ndarray
        Source point cloud, shape (3, N)
    Y : numpy.ndarray
        Target point cloud, shape (3, N)
    bound : float
        Noise bound
    length_bound : float
        Length bound for filtering (line vectors shorter than this are discarded
        because short edges are more sensitive to noise)
    
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
    X_lv, map = line_vectors(X, 1)
    Y_lv = line_vectors(Y, 0)
    
    D_xlv = np.sqrt(np.sum(X_lv**2, axis=0))
    D_ylv = np.sqrt(np.sum(Y_lv**2, axis=0))
    
    # Length filtering: discard short line vectors (noise-sensitive)
    # Short edges have high noise sensitivity in their length ratios
    if length_bound > 0:
        idx = (D_xlv <= length_bound) | (D_ylv <= length_bound)
    else:
        idx = np.zeros_like(D_xlv, dtype=bool)
    
    D_xlv = D_xlv[~idx]
    D_ylv = D_ylv[~idx]
    map = map[:, ~idx]
    X_lv = X_lv[:, ~idx]
    Y_lv = Y_lv[:, ~idx]
    
    eps = np.finfo(float).eps
    Sxy = D_ylv / (D_xlv + eps)
    Snoise = bound / (D_xlv + eps)
    
    return Sxy, Snoise, X_lv, Y_lv, map
