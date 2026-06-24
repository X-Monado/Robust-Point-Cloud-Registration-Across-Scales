import numpy as np


def build_pt_estimate(X, Y, ky_point, anchor_indices=None):
    """
    Build point estimate structure with optional anchor selection by degree.

    Parameters:
    -----------
    X : numpy.ndarray
        Source point cloud, shape (3, N)
    Y : numpy.ndarray
        Target point cloud, shape (3, N)
    ky_point : int
        Number of keypoints (anchors) to use
    anchor_indices : numpy.ndarray or None
        If provided, use these indices as anchors (e.g., from degree sorting).
        If None, fall back to sequential selection (first ky_point points).

    Returns:
    --------
    P : numpy.ndarray
        Point differences from source, shape (3, M)
    T : numpy.ndarray
        Point differences from target, shape (3, M)
    pt_map : numpy.ndarray
        Index map, shape (2, M)
    """
    x_size = X.shape[1]
    P = np.array([]).reshape(3, 0)
    T = np.array([]).reshape(3, 0)

    # Determine anchor indices
    if anchor_indices is None:
        # Fallback: sequential selection (original behavior)
        ky_point = min(ky_point, x_size)
        anchor_indices = np.arange(ky_point)
    else:
        ky_point = min(len(anchor_indices), ky_point, x_size)
        anchor_indices = anchor_indices[:ky_point]

    # Initialize point map
    pt_map = np.zeros((2, (x_size - 1) * ky_point), dtype=int)

    for ky_ind in range(1, ky_point + 1):
        ky_idx = int(anchor_indices[ky_ind - 1])  # Anchor point index

        # Front part (points before ky_idx)
        if ky_idx > 0:
            x_t = X[:, :ky_idx] - X[:, ky_idx:ky_idx+1]
            y_t = Y[:, :ky_idx] - Y[:, ky_idx:ky_idx+1]
            P = np.hstack([P, x_t])
            T = np.hstack([T, y_t])

            pre_ind = np.arange(ky_idx)
            len_c = len(pre_ind)
            if ky_ind != 1:
                pre_a = (ky_ind - 1) * (x_size - 1)
                pt_map[0, pre_a:pre_a + len_c] = pre_ind
        else:
            pre_a = 0
            len_c = 0

        # Back part (points after ky_idx)
        if ky_idx < x_size - 1:
            x_t = X[:, ky_idx+1:x_size] - X[:, ky_idx:ky_idx+1]
            y_t = Y[:, ky_idx+1:x_size] - Y[:, ky_idx:ky_idx+1]
            P = np.hstack([P, x_t])
            T = np.hstack([T, y_t])

            fanal_ind = np.arange(ky_idx + 1, x_size)
            pre_a = pre_a + len_c
            len_c = len(fanal_ind)
            pt_map[0, pre_a:pre_a + len_c] = fanal_ind

    return P, T, pt_map
