import numpy as np


def line_vectors(X, flag):
    """
    Generate line vectors between all pairs of points in X
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input point cloud, shape (3, N)
    flag : int
        If 1, also return the index map
    
    Returns:
    --------
    LV : numpy.ndarray
        Line vectors, shape (3, N*(N-1)/2)
    ID : numpy.ndarray, optional
        Index map, shape (2, N*(N-1)/2), only returned if flag == 1
    """
    N = X.shape[1]
    LV = np.zeros((X.shape[0], N * (N - 1) // 2))
    L = 0
    H = 0
    
    if flag == 1:
        ID = np.zeros((2, N * (N - 1) // 2), dtype=int)
    
    for i in range(N):
        x1 = X[:, i:i+1]
        x2 = X[:, i+1:N]
        SN = x2.shape[1]
        lv = np.tile(x1, (1, SN)) - x2
        H += SN
        
        if flag == 1:
            ID[0, L:H] = i
            ID[1, L:H] = np.arange(i+1, N)
        
        LV[:, L:H] = lv
        L = H
    
    if flag == 1:
        return LV, ID
    else:
        return LV
