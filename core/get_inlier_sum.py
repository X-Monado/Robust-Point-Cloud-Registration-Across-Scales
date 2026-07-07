import numpy as np


def get_inlier_sum(X, Y, R, t):
    """
    Compute the number of inliers given a transformation
    
    Parameters:
    -----------
    X : numpy.ndarray
        Source point cloud, shape (3, N)
    Y : numpy.ndarray
        Target point cloud, shape (3, N)
    R : numpy.ndarray
        Rotation matrix, shape (3, 3)
    t : numpy.ndarray
        Translation vector, shape (3,)
    
    Returns:
    --------
    cout_in : int
        Number of inliers
    """
    Y_out = np.dot(R, X) + t.reshape(-1, 1)
    errer_Y = Y - Y_out
    norm_2 = np.linalg.norm(errer_Y, axis=0)
    
    # Find points with error < 0.26
    f_in = np.where(norm_2 < 0.26)[0]
    cout_in = len(f_in)
    
    return cout_in
