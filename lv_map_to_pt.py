import numpy as np
from collections import Counter


def lv_map_to_pt(X, Y, map_arr, inliers, percent, distances):
    """
    Map line vectors back to points
    
    Parameters:
    -----------
    X : numpy.ndarray
        Source point cloud, shape (3, N)
    Y : numpy.ndarray
        Target point cloud, shape (3, N)
    map_arr : numpy.ndarray
        Index map from line_vectors, shape (2, M)
    inliers : numpy.ndarray
        Indices of inlier line vectors
    percent : float
        Percentage threshold for selecting points
    distances : numpy.ndarray
        Distances for each correspondence
    
    Returns:
    --------
    Xin : numpy.ndarray
        Selected source points, shape (3, K)
    Yin : numpy.ndarray
        Selected target points, shape (3, K)
    distance : numpy.ndarray
        Sorted distances, shape (K,)
    inlierPTS : numpy.ndarray
        Indices of selected points, shape (K,)
    """
    map_in = map_arr[:, inliers]
    
    # Count occurrences of each point index
    all_indices = map_in.flatten()
    counter = Counter(all_indices)
    
    # Sort points by count in descending order
    sorted_counts = sorted(counter.items(), key=lambda x: -x[1])
    sorted_indices = [item[0] for item in sorted_counts]
    sorted_values = [item[1] for item in sorted_counts]
    
    # Compute cumulative sum and find threshold
    cumsum = np.cumsum(sorted_values)
    total = cumsum[-1] if len(cumsum) > 0 else 0
    threshold = (percent / 100.0) * total if total > 0 else 0
    
    # Select points until cumulative count exceeds threshold
    p_idx = cumsum < threshold
    inlierPTS = np.array(sorted_indices)[p_idx]
    
    # Sort by distance (descending, since smaller feature distance correlates with larger GT error)
    if len(inlierPTS) > 0:
        distance_pre = distances[inlierPTS]
        # 确保 distance_pre 是一维数组
        if len(distance_pre.shape) == 2 and distance_pre.shape[1] == 2:
            distance_pre = distance_pre[:, 1]
        sorted_idx = np.argsort(-distance_pre)
        distance = distance_pre[sorted_idx]
        inlierPTS = inlierPTS[sorted_idx]
    else:
        distance = np.array([])
        inlierPTS = np.array([])
    
    # Get selected points
    Xin = X[:, inlierPTS] if len(inlierPTS) > 0 else np.array([])
    Yin = Y[:, inlierPTS] if len(inlierPTS) > 0 else np.array([])
    
    return Xin, Yin, distance, inlierPTS
