"""
Scale-Aware Triangular Consistency Constraint Module

This module implements the explicit three-condition check (C1, C2, C3) for
scale-aware triangular consistency as described in the paper.

For any correspondence pair (i, j) sharing a common anchor correspondence a,
the triangular constraint requires ALL three conditions to hold simultaneously:

    C1: | ||v_ay^Y|| - s * ||v_ax^Y|| | < epsilon_2
    C2: | ||v_ay^Y|| - s * ||v_ay^X|| | < epsilon_2
    C3: | ||v_ij^Y|| - s * ||v_ij^X|| | < epsilon_3

where v_ab = x_a - x_b is the line vector from a to b.
"""

import numpy as np


def compute_vertex_degrees(X, Y, s, epsilon_2, epsilon_3, anchor_idx):
    """
    Compute vertex degrees in the scale-aware triangular consistency graph.

    For each anchor a, we check for every pair (i, j) with i != a, j != a, i < j
    whether the three triangular conditions C1, C2, C3 hold. If they do, an edge
    between i and j is established (anchored at a). The degree of each vertex is
    the number of edges it participates in.

    Parameters:
    -----------
    X : numpy.ndarray, shape (3, N)
        Source point cloud
    Y : numpy.ndarray, shape (3, N)
        Target point cloud
    s : float
        Estimated scale factor
    epsilon_2 : float
        Compatibility threshold for anchor-to-point distances (C1, C2)
    epsilon_3 : float
        Compatibility threshold for point-to-point distance (C3)
    anchor_idx : int
        Index of the anchor point a

    Returns:
    --------
    degrees : numpy.ndarray, shape (N,)
        Degree of each vertex in the consistency graph anchored at anchor_idx
    edge_set : set of frozensets
        Set of edges (i, j) that satisfy all three conditions
    """
    N = X.shape[1]
    degrees = np.zeros(N, dtype=int)
    edge_set = set()

    # Anchor point
    x_a = X[:, anchor_idx]
    y_a = Y[:, anchor_idx]

    # Precompute anchor-to-point distances
    D_xa = np.linalg.norm(X - x_a.reshape(3, 1), axis=0)  # shape (N,)
    D_ya = np.linalg.norm(Y - y_a.reshape(3, 1), axis=0)  # shape (N,)

    # Precompute pairwise distances (source and target)
    # Using broadcasting: D_x[i,j] = ||x_i - x_j||
    D_x = np.linalg.norm(X[:, :, None] - X[:, None, :], axis=0)  # shape (N, N)
    D_y = np.linalg.norm(Y[:, :, None] - Y[:, None, :], axis=0)  # shape (N, N)

    # C1: |D_ya[i] - s * D_xa[i]| < epsilon_2 (anchor to i)
    c1_mask = np.abs(D_ya - s * D_xa) < epsilon_2  # shape (N,)

    # C2: |D_ya[j] - s * D_xa[j]| < epsilon_2 (anchor to j)
    c2_mask = c1_mask  # Same condition for j

    # C3: |D_y[i,j] - s * D_x[i,j]| < epsilon_3 (i to j)
    c3_mask = np.abs(D_y - s * D_x) < epsilon_3  # shape (N, N)

    # Combined mask: all three conditions must hold
    # For pair (i, j) with anchor a: c1_mask[i] AND c2_mask[j] AND c3_mask[i, j]
    for i in range(N):
        if i == anchor_idx or not c1_mask[i]:
            continue
        for j in range(i + 1, N):
            if j == anchor_idx or not c2_mask[j]:
                continue
            if c3_mask[i, j]:
                degrees[i] += 1
                degrees[j] += 1
                edge_set.add(frozenset({i, j}))

    return degrees, edge_set


def select_anchor_by_degree(X, Y, s, epsilon_2, epsilon_3, num_anchors=None):
    """
    Select anchor points based on degree ranking.

    The anchor selection strategy picks the point with the highest degree
    in the current consistency graph as the anchor for subsequent edge
    evaluations. This is motivated by the observation that high-degree
    vertices are more likely to be true inliers.

    Parameters:
    -----------
    X, Y : numpy.ndarray, shape (3, N)
        Point clouds
    s : float
        Scale factor
    epsilon_2, epsilon_3 : float
        Compatibility thresholds
    num_anchors : int or None
        Number of anchors to select. If None, uses min(150, N).

    Returns:
    --------
    anchor_indices : numpy.ndarray
        Indices of selected anchors, sorted by degree (descending)
    all_degrees : numpy.ndarray
        Degree of each vertex in the full consistency graph
    """
    N = X.shape[1]
    if num_anchors is None:
        num_anchors = min(150, N)
    else:
        num_anchors = min(num_anchors, N)

    # Use a fixed anchor (point 0) to compute initial degrees
    # This avoids O(N^2) anchor selection cost
    initial_anchor = 0
    degrees, _ = compute_vertex_degrees(X, Y, s, epsilon_2, epsilon_3, initial_anchor)

    # Sort vertices by degree (descending) and select top num_anchors
    sorted_indices = np.argsort(-degrees)
    anchor_indices = sorted_indices[:num_anchors]

    return anchor_indices, degrees


def build_triangular_consistency_graph(X, Y, s, epsilon_2, epsilon_3, anchor_indices=None):
    """
    Build the scale-aware triangular consistency graph.

    The graph has N vertices (one per correspondence). An edge (i, j) exists
    if there is an anchor a such that C1(i, a) AND C2(j, a) AND C3(i, j) hold.

    Parameters:
    -----------
    X, Y : numpy.ndarray, shape (3, N)
        Point clouds
    s : float
        Scale factor
    epsilon_2, epsilon_3 : float
        Compatibility thresholds
    anchor_indices : numpy.ndarray or None
        Indices of anchors to use. If None, selects anchors by degree.

    Returns:
    --------
    adjacency : numpy.ndarray, shape (N, N)
        Adjacency matrix of the consistency graph
    degrees : numpy.ndarray, shape (N,)
        Degree of each vertex
    """
    N = X.shape[1]

    if anchor_indices is None:
        anchor_indices, _ = select_anchor_by_degree(X, Y, s, epsilon_2, epsilon_3)

    adjacency = np.zeros((N, N), dtype=bool)
    degrees = np.zeros(N, dtype=int)

    # For each anchor, compute edges and accumulate
    for a in anchor_indices:
        _, edge_set = compute_vertex_degrees(X, Y, s, epsilon_2, epsilon_3, a)
        for edge in edge_set:
            i, j = sorted(edge)
            if not adjacency[i, j]:
                adjacency[i, j] = True
                adjacency[j, i] = True
                degrees[i] += 1
                degrees[j] += 1

    return adjacency, degrees


def filter_by_triangular_consistency(X, Y, s, epsilon_2, epsilon_3, percent=50):
    """
    Filter correspondences using scale-aware triangular consistency.

    This implements the full pipeline:
    1. Build the triangular consistency graph
    2. Compute vertex degrees
    3. Select top vertices by cumulative frequency (default: top 50%)

    Parameters:
    -----------
    X, Y : numpy.ndarray, shape (3, N)
        Point clouds
    s : float
        Scale factor
    epsilon_2, epsilon_3 : float
        Compatibility thresholds
    percent : float
        Cumulative frequency threshold (0-100)

    Returns:
    --------
    inlier_indices : numpy.ndarray
        Indices of filtered inliers
    degrees : numpy.ndarray
        Degree of each vertex
    """
    from collections import Counter

    N = X.shape[1]

    # Build graph and compute degrees
    _, degrees = build_triangular_consistency_graph(
        X, Y, s, epsilon_2, epsilon_3, anchor_indices=None
    )

    # Sort by degree (descending)
    sorted_indices = np.argsort(-degrees)
    sorted_degrees = degrees[sorted_indices]

    # Cumulative frequency
    cumsum = np.cumsum(sorted_degrees)
    total = cumsum[-1] if len(cumsum) > 0 else 0
    threshold = (percent / 100.0) * total if total > 0 else 0

    # Select vertices contributing to top `percent`% cumulative frequency
    p_idx = cumsum < threshold
    inlier_indices = sorted_indices[p_idx]

    return inlier_indices, degrees
