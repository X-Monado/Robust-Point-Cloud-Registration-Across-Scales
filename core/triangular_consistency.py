"""
Scale-Aware Triangular Consistency Constraint Module (Vectorized)

For any correspondence pair (i, j) sharing a common anchor correspondence a,
the triangular constraint requires ALL three conditions to hold simultaneously:

    C1: | ||v_ai^Y|| - s * ||v_ai^X|| | < epsilon_2
    C2: | ||v_aj^Y|| - s * ||v_aj^X|| | < epsilon_2
    C3: | ||v_ij^Y|| - s * ||v_ij^X|| | < epsilon_3

where v_ab = x_a - x_b is the line vector from a to b.
"""

import numpy as np


def compute_vertex_degrees(X, Y, s, epsilon_2, epsilon_3, anchor_idx):
    """
    Compute vertex degrees using vectorized operations.

    Parameters:
    -----------
    X, Y : numpy.ndarray, shape (3, N)
    s : float - Scale factor
    epsilon_2, epsilon_3 : float - Compatibility thresholds
    anchor_idx : int - Index of anchor point

    Returns:
    --------
    degrees : numpy.ndarray, shape (N,)
    edge_mask : numpy.ndarray, shape (N, N) - Upper triangular edge mask
    """
    N = X.shape[1]

    x_a = X[:, anchor_idx]
    y_a = Y[:, anchor_idx]

    # Anchor-to-point distances
    D_xa = np.linalg.norm(X - x_a.reshape(3, 1), axis=0)
    D_ya = np.linalg.norm(Y - y_a.reshape(3, 1), axis=0)

    # C1/C2: |D_ya - s * D_xa| < epsilon_2
    c12_mask = np.abs(D_ya - s * D_xa) < epsilon_2

    # Pairwise distances via broadcasting
    D_x = np.linalg.norm(X[:, :, None] - X[:, None, :], axis=0)
    D_y = np.linalg.norm(Y[:, :, None] - Y[:, None, :], axis=0)

    # C3: |D_y - s * D_x| < epsilon_3
    c3_mask = np.abs(D_y - s * D_x) < epsilon_3

    # Combined: c12_mask[i] AND c12_mask[j] AND c3_mask[i,j]
    c12_outer = np.outer(c12_mask, c12_mask)
    np.fill_diagonal(c12_outer, False)
    c12_outer[anchor_idx, :] = False
    c12_outer[:, anchor_idx] = False

    edge_mask = c12_outer & c3_mask
    edge_mask_upper = np.triu(edge_mask, k=1)

    degrees = np.sum(edge_mask_upper, axis=0) + np.sum(edge_mask_upper, axis=1)

    return degrees, edge_mask_upper


def select_anchor_by_degree(X, Y, s, epsilon_2, epsilon_3, num_anchors=None):
    """
    Select anchor points based on degree ranking.
    Uses a single initial anchor, then selects top-K by degree.
    """
    N = X.shape[1]
    if num_anchors is None:
        num_anchors = min(10, N)
    else:
        num_anchors = min(num_anchors, N)

    # Use point 0 as initial anchor
    degrees, _ = compute_vertex_degrees(X, Y, s, epsilon_2, epsilon_3, 0)
    sorted_indices = np.argsort(-degrees)
    anchor_indices = sorted_indices[:num_anchors]

    return anchor_indices, degrees


def build_triangular_consistency_graph(X, Y, s, epsilon_2, epsilon_3, anchor_indices=None):
    """
    Build the scale-aware triangular consistency graph.
    Uses multiple anchors and accumulates edges.
    """
    N = X.shape[1]

    if anchor_indices is None:
        anchor_indices, _ = select_anchor_by_degree(X, Y, s, epsilon_2, epsilon_3)

    adjacency = np.zeros((N, N), dtype=bool)
    degrees = np.zeros(N, dtype=int)

    for a in anchor_indices:
        _, edge_mask = compute_vertex_degrees(X, Y, s, epsilon_2, epsilon_3, a)
        # Accumulate new edges
        new_edges = edge_mask & ~adjacency
        adjacency |= new_edges
        degrees += np.sum(new_edges, axis=0) + np.sum(new_edges, axis=1)

    return adjacency, degrees


def filter_by_triangular_consistency(X, Y, s, epsilon_2, epsilon_3, percent=50):
    """
    Filter correspondences using scale-aware triangular consistency.

    Parameters:
    -----------
    X, Y : numpy.ndarray, shape (3, N)
    s : float - Scale factor
    epsilon_2, epsilon_3 : float - Compatibility thresholds
    percent : float - Cumulative frequency threshold (0-100)

    Returns:
    --------
    inlier_indices : numpy.ndarray - Indices of filtered inliers
    degrees : numpy.ndarray - Degree of each vertex
    """
    N = X.shape[1]

    _, degrees = build_triangular_consistency_graph(
        X, Y, s, epsilon_2, epsilon_3, anchor_indices=None
    )

    # Sort by degree (descending)
    sorted_indices = np.argsort(-degrees)
    sorted_degrees = degrees[sorted_indices]

    # Cumulative frequency
    cumsum = np.cumsum(sorted_degrees)
    total = cumsum[-1] if len(cumsum) > 0 and cumsum[-1] > 0 else 0

    if total == 0:
        # No edges found, return all indices
        return np.arange(N), degrees

    threshold = (percent / 100.0) * total
    p_idx = cumsum < threshold

    # Ensure at least a few inliers
    if np.sum(p_idx) < 3:
        p_idx[:min(5, N)] = True

    inlier_indices = sorted_indices[p_idx]

    return inlier_indices, degrees
